/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
template <>
LazyGeometric<at::Tensor&>::LazyGeometric(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  // Generators can't be represented in JIT graph
  // https://github.com/pytorch/pytorch/issues/64005
  // Seed is always at the end for all variants
  get_inputs().back() = static_cast<int64_t>(
      get_seed_hpu(inputs.back().toOptional<at::Generator>()));
}

template <>
at::Tensor& LazyGeometric<at::Tensor&>::get_result_overrideable() {
  return stack_tensor(get_inputs(), 0);
}

std::shared_ptr<void> FillRandomNegativeBinomialParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomNegativeBinomial::Params);
  auto self = stack.at(0).toTensor();
  auto p = stack.at(1).toScalar().to<float>();
  auto seed = stack.at(2).toInt();

  params->p = p;
  params->k = 1.0;
  params->seed = seed;
  return params;
}

void Geometric::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  size_t size = 0;
  const auto& params = FillRandomNegativeBinomialParams(stack, size);

  auto random_neg_binomial = BuildOp(
      graph,
      "random_negative_binomial_fwd_" +
          habana_helpers::name_suffix_from_type(ScalarType()),
      {},
      {{outshape, ScalarType(), false}},
      params.get(),
      size);

  auto constant = ConstantHelper(graph, 1, ScalarType(), outshape);

  /* The geometric distribution Y is a special case of the negative binomial
   * distribution, with k = 1. random_negative_binomial returns distributions as
   * number of failures, but number of trials is expected by pytorch. Number of
   * trials in case of k=1 is number of failures + 1 (success).*/
  auto geometric = BuildOp(
      graph,
      "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {random_neg_binomial[0].get(), constant.get()},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}});

  syn_out(0) = std::move(geometric[0]);
}
} // namespace habana

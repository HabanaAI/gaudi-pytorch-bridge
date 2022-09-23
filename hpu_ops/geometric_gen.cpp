/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/geometric.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
template <>
LazyGeometric<at::Tensor&>::LazyGeometric(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  // Generators can't be represented in JIT graph
  // https://github.com/pytorch/pytorch/issues/64005
  LazyGeometric<at::Tensor&>::get_inputs().at(2) =
      get_seed_tensor_hpu(inputs.at(2).toOptional<at::Generator>());
}

template <>
at::Tensor& LazyGeometric<at::Tensor&>::get_result_overrideable() {
  return stack_tensor(LazyGeometric<at::Tensor&>::get_inputs(), 0);
}

std::shared_ptr<void> FillRandomNegativeBinomialParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomNegativeBinomial::ParamsV2);
  auto self = stack.at(0).toTensor();
  auto p = stack.at(1).toScalar().to<float>();

  params->p = p;
  params->k = 1.0;
  params->isAdditionEnable = true;

  return params;
}

void Geometric::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  size_t size = 0;
  const auto& params = FillRandomNegativeBinomialParams(stack, size);

  auto geometric = BuildOp(
      graph,
      "random_negative_binomial_fwd_" +
          habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1)},
      {{outshape, ScalarType(), 0}},
      params.get(),
      size);

  syn_out(0) = std::move(geometric[0]);
}
} // namespace habana

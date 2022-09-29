/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/exponential.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {

template <typename T>
LazyTensorOutSeedExp<T>::LazyTensorOutSeedExp(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes_fn) {
  // Generators can't be represented in JIT graph
  // https://github.com/pytorch/pytorch/issues/64005
  LazyTensorOutSeedExp<T>::get_inputs().at(2) =
      get_seed_tensor_hpu(inputs.at(2).toOptional<at::Generator>());
}

template <typename T>
T LazyTensorOutSeedExp<T>::get_result_overrideable() {
  return stack_tensor(LazyTensorOutSeedExp<T>::get_inputs(), 0);
}

template struct LazyTensorOutSeedExp<at::Tensor&>;
template struct LazyTensorOutSeedExp<at::Tensor>;

std::shared_ptr<void> FillExponentialParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomExponential::Params);
  float lambd = stack.at(1).toScalar().toFloat();
  TORCH_CHECK(
      lambd >= 0.0,
      "exponential_ expects lambda >= 0.0, but found lambda=",
      lambd);
  params->beta = 1.0f / lambd;
  return params;
}

void ExponentialSeedTensorInput::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  // Discard self tensor, input is seed tensor only
  p_context_->syn_inputs_.pop_front();
  HABANA_ASSERT(p_context_->syn_inputs_.size() == 1);

  auto outshape = stack_tensor(stack, 0).sizes();
  size_t size = 0;
  auto params = FillExponentialParams(stack, size);

  std::vector<synTensor> inputs = {syn_in(0)};
  CreateShapeTensorInput(graph, ScalarType(), outshape, inputs);

  auto exponential = BuildOp(
      graph,
      "random_exponential_fwd_" +
          habana_helpers::name_suffix_from_type(ScalarType()),
      inputs,
      {{outshape, ScalarType(), 0}},
      params.get(),
      size);
  syn_out(0) = std::move(exponential[0]);
}
} // namespace habana

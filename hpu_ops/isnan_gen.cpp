/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/isnan.h"

namespace habana {

template <>
LazyIsNan<at::Tensor>::LazyIsNan(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
at::Tensor LazyIsNan<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(0).toTensor();
  const auto& options = t.options().dtype(torch::kBool);
  return habana_lazy::empty_hpu_lazy(
      t.sizes(), options, t.suggest_memory_format(), false);
}

sizes_vec IsNanOutputShape(const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() == 1,
      "Incorrect number of inputs provided, while expected 1 input for Isnan");
  TORCH_CHECK(
      stack.at(0).isTensor(), "Input arg1 expected to be Tensor for Isnan");

  const torch::Tensor& self = stack_tensor(stack, 0);
  return {self.sizes().vec()};
}

void IsNanOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  auto outshape = IsNanOutputShape(stack)[0];

  if (c10::isFloatingType(self.scalar_type())) {
    auto result = BuildOp(
        graph,
        "isnan_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0)},
        {{outshape, torch::kBool, 0}});
    syn_out(0) = std::move(result[0]);
  } else {
    auto cast_to_f32 = CastHelper(
        graph, syn_in(0), outshape, self.scalar_type(), c10::ScalarType::Float);
    auto result = BuildOp(
        graph,
        "isnan_fwd_f32",
        {cast_to_f32.get()},
        {{outshape, torch::kBool, 0}});
    syn_out(0) = std::move(result[0]);
  }
}

} // namespace habana

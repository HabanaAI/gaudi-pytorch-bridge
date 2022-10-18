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
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  set_scalar_type(torch::kBool);
}

template <>
at::Tensor LazyIsNan<at::Tensor>::get_result_overrideable() {
  return {};
}

void IsNanOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const at::Tensor& self = stack_tensor(stack, 0);
  const auto& outshape = self.sizes();

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

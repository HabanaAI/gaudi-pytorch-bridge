/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/fill.h"

namespace habana {
template <>
FillFE<at::Tensor&>::FillFE(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  // NOTE: Workaround due to https://github.com/pytorch/pytorch/issues/75465
  if (inputs[1].isBool()) {
    get_inputs()[1] = static_cast<int>(inputs[1].toBool());
  }
}

template <>
at::Tensor& FillFE<at::Tensor&>::get_result_overrideable() {
  return habana_lazy::LazyOp<at::Tensor&>::get_result_overrideable();
}

void Fill::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  auto broadcast = BuildOp(
      graph,
      "broadcast_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1)},
      {{outshape, ScalarType(), 0}});

  // output of broadcast is the output of this op
  syn_out(0) = std::move(broadcast[0]);
}

void FillScalar::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto other = stack.at(1).toScalar();
  const auto& outshape = self.sizes();
  auto result = ConstantHelper(graph, other, ScalarType(), outshape, 0);
  syn_out(0) = std::move(result);
}

} // namespace habana

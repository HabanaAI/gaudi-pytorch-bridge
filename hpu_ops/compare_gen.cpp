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

namespace habana {
template <>
LazyCmp<at::Tensor>::LazyCmp(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
at::Tensor LazyCmp<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(0).toTensor();
  return habana_lazy::empty_hpu_lazy(
      t.sizes(),
      t.options().dtype(at::kBool),
      t.suggest_memory_format(),
      false);
}

void CompareOp::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  auto outshape = BinaryOutputShape(stack)[0];
  auto result = BuildOp(
      graph,
      guid_,
      {syn_in(0), syn_in(1)},
      {{outshape, at::kBool, is_output_persistent_list[0], true}});

  syn_out(0) = std::move(result[0]);
}
} // namespace habana

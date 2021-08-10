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
void RsubOp::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  if (ScalarInputs().at(ScalarId()).toFloat() == 1.) {
    p_context_->syn_inputs_.erase(
        p_context_->syn_inputs_.cbegin() + ScalarId());
    std::swap(syn_in(0), syn_in(1));
    return HabanaOperatorHelper::AddNode(
        graph, stack, is_output_persistent_list);
  }
  auto mul = BuildOp(
      graph,
      "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), syn_in(2)},
      {{stack_tensor(stack, 1).sizes(), ScalarType(), false}});

  auto op = BuildOp(
      graph,
      guid_,
      {syn_in(1), mul[0].get()},
      {{stack_tensor(stack, 0).sizes(),
        ScalarType(),
        is_output_persistent_list[0],
        IsOutFn() ? 0 : -1}});

  syn_out(0) = std::move(op[0]);
}
} // namespace habana

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
void BinaryWithAlphaOutOp::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    bool is_output_persistent) {
  if (ScalarInputs().at(ScalarId()).toFloat() == 1.) {
    p_context_->syn_inputs_.erase(
        p_context_->syn_inputs_.cbegin() + ScalarId());
    return HabanaOperatorHelper::AddNode(graph, stack, is_output_persistent);
  }

  auto mul = BuildOp(
      "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      graph,
      {syn_in(1), syn_in(2)},
      {{stack_tensor(stack, 1).sizes(), ScalarType(), false}});

  auto op = BuildOp(
      guid_,
      graph,
      {syn_in(0), mul[0].get()},
      {{stack_tensor(stack, 0).sizes(), ScalarType(), is_output_persistent}});

  syn_out(0) = std::move(op[0]);
}
} // namespace habana

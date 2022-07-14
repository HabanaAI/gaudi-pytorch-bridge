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

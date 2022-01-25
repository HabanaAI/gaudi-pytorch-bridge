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

} // namespace habana

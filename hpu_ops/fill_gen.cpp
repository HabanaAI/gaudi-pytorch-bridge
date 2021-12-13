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
void Fill::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  auto broadcast = BuildOp(
      graph,
      "broadcast_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1)},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}});

  // output of broadcast is the output of this op
  syn_out(0) = std::move(broadcast[0]);
}

} // namespace habana

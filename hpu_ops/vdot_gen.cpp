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
sizes_vec VdotOutputShape(const at::Stack&, bool) {
  return {{}};
}

void VdotOperator::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  auto mat1 = stack.at(0).toTensor();
  auto mat2 = stack.at(1).toTensor();

  auto reshape_m1 = BuildOp(
      graph, "reshape", {syn_in(0)}, {{{1, mat1.numel()}, ScalarType()}});

  auto reshape_m2 = BuildOp(
      graph, "reshape", {syn_in(1)}, {{{mat2.numel(), 1}, ScalarType()}});

  auto mm = BuildOp(
      graph,
      "gemm",
      {reshape_m1[0].get(), reshape_m2[0].get()},
      {{{1, 1}, ScalarType()}});

  auto vdot = BuildOp(
      graph,
      "reshape",
      {mm[0].get()},
      {{1, ScalarType(), is_output_persistent_list[0], true}});
  syn_out(0) = std::move(vdot[0]);
}
} // namespace habana

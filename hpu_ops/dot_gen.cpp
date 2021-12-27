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
#include "hpu_op_helper.h"

namespace habana {
sizes_vec DotOutputShape(const at::Stack& stack, bool) {
  const at::Tensor self = stack_tensor(stack, 0);
  const at::Tensor other = stack_tensor(stack, 1);
  TORCH_CHECK(
      self.dim() == 1 && other.dim() == 1,
      "Dot Op: 1D tensors expected, but got ",
      self.dim(),
      "D and ",
      other.dim(),
      "D tensors");
  TORCH_CHECK(
      self.sizes() == other.sizes(),
      "Dot Op: Tensor must have same size, but got ",
      self.sizes(),
      "and ",
      other.sizes(),
      "size tensors");
  return {{}};
}

void Dot::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  auto mat1 = stack.at(0).toTensor();
  auto mat2 = stack.at(1).toTensor();

  auto reshape_m1 = BuildOp(
      graph, "reshape", {syn_in(0)}, {{{1, mat1.numel()}, ScalarType()}});

  auto reshape_m2 = BuildOp(
      graph, "reshape", {syn_in(1)}, {{{mat2.numel(), 1}, ScalarType()}});

  // gemm supports only Float32 and BFloat16
  // Issue raised : https://jira.habana-labs.com/browse/SW-69290
  auto mm = BuildOp(
      graph,
      "gemm",
      {reshape_m1[0].get(), reshape_m2[0].get()},
      {{{1, 1}, ScalarType()}});

  auto dot = BuildOp(
      graph,
      "reshape",
      {mm[0].get()},
      {{1, ScalarType(), is_output_persistent_list[0], true}});
  syn_out(0) = std::move(dot[0]);
}
} // namespace habana

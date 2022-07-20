/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/mv.h"

namespace habana {

sizes_vec MvOpsOutputShape(const at::Stack& stack, bool) {
  const at::Tensor mat1 = stack_tensor(stack, 0);
  sizes_vec shape = std::vector<std::vector<int64_t>>{{mat1.sizes()[0]}};
  return shape;
}
void MvOp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const at::Tensor mat1 = stack_tensor(stack, 0);
  const at::Tensor mat2 = stack_tensor(stack, 1);

  int64_t data_1[] = {mat2.numel(), 1};
  c10::IntArrayRef shape_1(data_1, 2);

  auto reshapeOp = ReshapeHelper(graph, syn_in(1), shape_1, ScalarType());

  int64_t data_2[] = {mat1.sizes()[0], 1};
  c10::IntArrayRef shape_2(data_2, 2);

  auto mmOp = BuildOp(
      graph, "gemm", {syn_in(0), reshapeOp.get()}, {{shape_2, ScalarType()}});

  int64_t data_3[] = {mat1.sizes()[0]};
  c10::IntArrayRef shape_3(data_3, 1);

  auto reshapeOp2 =
      ReshapeHelper(graph, mmOp[0].get(), shape_3, ScalarType(), 0);

  // output
  syn_out(0) = std::move(reshapeOp2);
}

} // namespace habana

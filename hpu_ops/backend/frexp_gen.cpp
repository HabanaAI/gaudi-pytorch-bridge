/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "generated/backend/frexp.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

sizes_vec FrexpOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> shape = self.sizes().vec();
  return {{shape, shape}};
}

void Frexp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = FrexpOutputShape(stack)[0];
  auto frexp = BuildOp(
      graph,
      guid_,
      {syn_in(0)},
      {{outshape, c10::ScalarType::Int, 1}, {outshape, ScalarType(), 0}});

  syn_out(0) = std::move(frexp[1]);
  syn_out(1) = std::move(frexp[0]);
}

} // namespace habana

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

#include "generated/backend/special_xlog1py.h"
#include "generated/backend/xlogy.h"

namespace habana {

sizes_vec XlogYOutputShape(const at::Stack& stack) {
  if (stack.at(1).isScalar()) {
    const torch::Tensor& self = stack_tensor(stack, 0);
    return {self.sizes().vec()};
  } else if (stack.at(0).isScalar()) {
    const torch::Tensor& other = stack_tensor(stack, 1);
    return {other.sizes().vec()};
  }
  const torch::Tensor& self = stack_tensor(stack, 0);
  const torch::Tensor& other = stack_tensor(stack, 1);
  return {at::infer_size(self.sizes(), other.sizes())};
}

void XlogYOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto outshape = XlogYOutputShape(stack)[0];
  auto other_shape = stack_tensor(stack, 1).sizes().vec();

  auto logy = BuildOp(graph, guid_, {syn_in(1)}, {{other_shape, ScalarType()}});
  auto xlogy = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), logy[0].get()},
      {{outshape, ScalarType(), 0}});

  syn_out(0) = std::move(xlogy[0]);
}
} // namespace habana

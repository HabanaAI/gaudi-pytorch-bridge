/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/linear_backward.h"

namespace habana {
sizes_vec ComputeLinearBwdShape(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);
  const auto& weight = stack_tensor(stack, 2);
  std::vector<int64_t> bias_grad_shape(1, weight.sizes().vec()[0]);

  return {input.sizes().vec(), weight.sizes().vec(), bias_grad_shape};
}

void LinearBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  std::swap(p_context_->syn_inputs_[0], p_context_->syn_inputs_[1]);
  return OpBackend::AddNode(graph, stack);
}
} // namespace habana

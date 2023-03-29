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
#include "generated/backend/masked_fill.h"

namespace habana {

void MaskedFill::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto value = stack.at(2);

  if (value.isScalar()) {
    p_context_->syn_inputs_.emplace_back(
        ConstantHelper(graph, value.toScalar(), self.scalar_type()));
  }

  auto result = BuildOp(
      graph,
      guid_,
      {syn_in(1), syn_in(2), syn_in(0)},
      {{self.sizes(), ScalarType(), 0}});

  syn_out(0) = std::move(result[0]);
}
} // namespace habana

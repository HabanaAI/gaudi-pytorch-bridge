/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/isfinite.h"
#include "generated/backend/isinf.h"

namespace habana {

void _IsFiniteInf::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  auto params = FillParams(stack, size);
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto result = BuildOp(
      graph,
      guid_,
      {syn_in(0)},
      {{outshape, torch::kBool, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(result[0]);
}

} // namespace habana

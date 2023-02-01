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

#include "generated/backend/isnan.h"

namespace habana {

void IsNanOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const at::Tensor& self = stack_tensor(stack, 0);
  const auto& outshape = self.sizes();

  if (c10::isFloatingType(self.scalar_type())) {
    auto result = BuildOp(
        graph,
        "isnan_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0)},
        {{outshape, torch::kBool, 0}});
    syn_out(0) = std::move(result[0]);
  } else {
    auto false_val = ConstantHelper(graph, false, at::kBool, outshape, 0);
    syn_out(0) = std::move(false_val);
  }
}

} // namespace habana

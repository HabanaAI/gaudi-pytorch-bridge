/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/ceil.h"
#include "generated/backend/floor.h"
#include "generated/backend/trunc.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {
SharedMetaDataVector RoundingTruncSharedMeta(const at::Stack& stack) {
  return RoundingSharedMeta(stack, "trunc_fwd");
}

SharedMetaDataVector RoundingCeilSharedMeta(const at::Stack& stack) {
  return RoundingSharedMeta(stack, "ceil_fwd");
}

SharedMetaDataVector RoundingFloorSharedMeta(const at::Stack& stack) {
  return RoundingSharedMeta(stack, "floor_fwd");
}

void RoundingFunc::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  std::string guid =
      c10::isIntegralType(ScalarType(), true) ? "identity" : guid_;
  auto result =
      BuildOp(graph, guid, {syn_in(0)}, {{outshape, ScalarType(), 0}});
  syn_out(0) = std::move(result[0]);
}

} // namespace habana

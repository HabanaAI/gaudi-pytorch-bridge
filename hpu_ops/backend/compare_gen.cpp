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

#include "generated/backend/eq.h"
#include "generated/backend/ge.h"
#include "generated/backend/gt.h"
#include "generated/backend/le.h"
#include "generated/backend/lt.h"
#include "generated/backend/ne.h"

namespace habana {

void CompareOp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = BinaryOutputShape(stack)[0];
  auto result =
      BuildOp(graph, guid_, {syn_in(0), syn_in(1)}, {{outshape, at::kBool, 0}});

  syn_out(0) = std::move(result[0]);
}
} // namespace habana

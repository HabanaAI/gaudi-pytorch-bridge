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

#include "generated/backend/div.h"

namespace habana {

void Divide::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  if (ScalarType() == at::ScalarType::Float) {
    // Update the div guid with precise based on env
    update_div_guid_with_precise(guid_);
  }
  return OpBackend::AddNode(graph, stack);
}
} // namespace habana

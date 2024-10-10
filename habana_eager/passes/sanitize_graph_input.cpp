/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************/

#include <c10/util/ArrayRef.h>

#include "habana_eager/graph_exec.h"
#include "habana_helpers/logging_pt.h"

namespace habana {
namespace graph {
namespace pass {

void SanitizeGraphInput(std::shared_ptr<torch::jit::Graph> graph) {
  PT_EAGER_TRACE;
  if (0 == graph->inputs().size()) {
    // No input to sanitize...
    return;
  }

  torch::jit::Value* first_graph_input{*graph->inputs().begin()};
  if (!first_graph_input->hasUses() &&
      "self" == first_graph_input->debugName()) {
    graph->eraseInput(0);
  }
}

} // namespace pass
} // namespace graph
} // namespace habana
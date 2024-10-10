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
#include <torch/csrc/jit/ir/ir.h>
#include "habana_eager/graph_exec.h"
#include "habana_helpers/logging_pt.h"

namespace habana {
namespace graph {
namespace pass {

bool RemoveDetachOp(std::shared_ptr<torch::jit::Graph> graph) {
  PT_EAGER_TRACE;
  auto nodes = graph->nodes();

  std::unordered_set<torch::jit::Node*> detach_nodes;
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    auto node = *it;
    if (node->kind() != torch::jit::aten::detach) {
      continue;
    }

    detach_nodes.insert(node);
  }

  for (auto n : detach_nodes) {
    n->output(0)->replaceAllUsesWith(n->input(0));
    n->destroy();
  }

  if (!detach_nodes.empty()) {
    PT_EAGER_INFO(__PRETTY_FUNCTION__, ": \n", *graph);
  }

  return !detach_nodes.empty();
}

} // namespace pass
} // namespace graph
} // namespace habana

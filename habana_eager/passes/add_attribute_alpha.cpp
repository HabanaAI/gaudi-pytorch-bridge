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
 *******************************************************************************
 */

#include <c10/util/ArrayRef.h>

#include "habana_eager/graph_exec.h"
#include "habana_helpers/logging_pt.h"

namespace habana {
namespace graph {
namespace pass {

struct AddAttributeAlphaPass {
  explicit AddAttributeAlphaPass(std::shared_ptr<torch::jit::Graph> graph)
      : m_graph(std::move(graph)) {}

  bool run() {
    if (!GET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE)) {
      return false;
    }
    return processBlocks(m_graph->block());
  }

 private:
  bool processBlocks(at::ArrayRef<torch::jit::Block*> blocks) {
    bool changed{false};
    auto& gconfig{HPURegistrar::get_hpu_global_config()};

    for (auto block : blocks) {
      for (auto node : block->nodes()) {
        changed |= processNode(node, gconfig.getDeterministic());
      }
    }
    return changed;
  }

  bool processNode(torch::jit::Node* node, bool deterministic) {
    auto one = torch::jit::attr::deterministic;
    node->i_(one, deterministic);
    return true;
  }

  std::shared_ptr<torch::jit::Graph> m_graph;
};

void AddAttributeAlpha(std::shared_ptr<torch::jit::Graph> graph) {
  PT_EAGER_TRACE;
  AddAttributeAlphaPass pass{graph};
  bool changed{pass.run()};
  if (changed) {
    PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
  }
}

} // namespace pass
} // namespace graph
} // namespace habana

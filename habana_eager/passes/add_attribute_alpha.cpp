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
    synapse_helpers::device& device{
        synapse_helpers::HPURegistrar::get_device()};

    for (auto block : blocks) {
      for (auto node : block->nodes()) {
        changed |= processNode(node, device);
      }
    }
    return changed;
  }

  bool processNode(torch::jit::Node* node, synapse_helpers::device& device) {
    auto one = torch::jit::attr::alpha;
    node->i_(one, device.getDeterministic());
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
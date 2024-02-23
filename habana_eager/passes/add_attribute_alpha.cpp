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
#include "backend/habana_device/hpu_cached_devices.h"
#include "habana_helpers/logging_pt.h" // Required for logging

namespace habana {
namespace graph {
namespace pass {

struct AddAttributeAlphaPass {
  explicit AddAttributeAlphaPass(std::shared_ptr<torch::jit::Graph> graph)
      : m_graph(std::move(graph)) {}

  bool run() {
    return processBlocks(m_graph->block());
  }

 private:
  bool processBlocks(at::ArrayRef<torch::jit::Block*> blocks) {
    bool changed{false};
    const auto deterministic =
        HPURegistrar::get_hpu_global_config().getDeterministic() ||
        at::globalContext().deterministicAlgorithms();

    for (auto block : blocks) {
      for (auto node : block->nodes()) {
        changed |= processNode(node, deterministic);
      }
    }

    return changed;
  }

  bool processNode(torch::jit::Node* node, bool deterministic) {
    node->i_(torch::jit::attr::deterministic, deterministic);
    return true;
  }

  std::shared_ptr<torch::jit::Graph> m_graph;
};

bool AddAttributeAlpha(std::shared_ptr<torch::jit::Graph> graph) {
  PT_EAGER_TRACE;
  AddAttributeAlphaPass pass{graph};
  bool changed{pass.run()};
  if (changed) {
    PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
  }
  return changed;
}

} // namespace pass
} // namespace graph
} // namespace habana

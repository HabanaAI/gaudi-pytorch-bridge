/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
    const auto deterministic = HPUGlobalConfig::get().getDeterministic() ||
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

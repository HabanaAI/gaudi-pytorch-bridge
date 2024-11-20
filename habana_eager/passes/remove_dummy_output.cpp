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
#include <torch/csrc/jit/ir/ir.h>
#include "habana_helpers/logging_pt.h"

namespace habana {
namespace graph {
namespace pass {
struct RemoveDummyOutputPass {
  explicit RemoveDummyOutputPass(std::shared_ptr<torch::jit::Graph> graph)
      : m_graph(std::move(graph)) {}
  bool run() {
    auto outputs = m_graph->outputs();
    if (outputs.size() == 1) {
      auto output_node = outputs[0]->node();
      if (output_node->kind() == torch::jit::prim::Constant) {
        HABANA_ASSERT(
            outputs[0]->uses().size() == 1,
            "Output node can be removed when it is not used elsewhere.");
        m_graph->eraseOutput(0);
        output_node->destroy();
        return true;
      }
    }
    return false;
  }

 private:
  std::shared_ptr<torch::jit::Graph> m_graph;
};

bool RemoveDummyOutput(std::shared_ptr<torch::jit::Graph> graph) {
  PT_EAGER_TRACE;
  RemoveDummyOutputPass pass{graph};
  bool changed{pass.run()};
  if (changed) {
    PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
  }
  return changed;
}

} // namespace pass
} // namespace graph
} // namespace habana

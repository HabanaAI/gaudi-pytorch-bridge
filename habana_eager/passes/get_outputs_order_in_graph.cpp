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

#include <algorithm>

#include <c10/util/ArrayRef.h>

#include <torch/csrc/jit/ir/ir.h>
#include "backend/jitgraph_utils.h"
#include "habana_eager/graph_exec.h"
#include "habana_helpers/logging_pt.h"

namespace habana {
namespace graph {
namespace pass {

struct GetOutputsOrderInGraphPass {
  explicit GetOutputsOrderInGraphPass(std::shared_ptr<torch::jit::Graph> graph)
      : m_graph(std::move(graph)) {}

  void run() {
    auto graph_outputs = m_graph->outputs();

    for (auto* node : m_graph->nodes()) {
      c10::ArrayRef<torch::jit::Value*> node_outputs = node->outputs();
      for (auto* node_output : node_outputs) {
        if (jitgraph_utils::isInGraphOutputs(node_output)) {
          at::ArrayRef<torch::jit::Value*>::iterator itr = std::find(
              graph_outputs.begin(), graph_outputs.end(), node_output);
          if (itr != graph_outputs.cend()) {
            int index = std::distance(graph_outputs.begin(), itr);
            m_outputs_order.push_back(index);
          }
        }
      }
    }
    m_graph->block()->permuteOutputs(m_outputs_order);
  }

  std::vector<size_t> get_outputs_order() {
    return m_outputs_order;
  }

 private:
  std::shared_ptr<torch::jit::Graph> m_graph;
  std::vector<size_t> m_outputs_order;
};

bool GetOutputsOrderInGraph(
    std::shared_ptr<torch::jit::Graph> graph,
    std::vector<size_t>& outputs_order) {
  PT_EAGER_TRACE;
  GetOutputsOrderInGraphPass pass{graph};
  pass.run();
  outputs_order = pass.get_outputs_order();
  return !std::is_sorted(outputs_order.begin(), outputs_order.end());
}

} // namespace pass
} // namespace graph
} // namespace habana

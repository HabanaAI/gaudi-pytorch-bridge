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

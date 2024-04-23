/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************/
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

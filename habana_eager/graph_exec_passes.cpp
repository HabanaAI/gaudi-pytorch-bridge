/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "habana_helpers/logging.h"

namespace habana {
namespace graph {
namespace pass {

struct HandleTupleOnOutputPass {
  explicit HandleTupleOnOutputPass(std::shared_ptr<torch::jit::Graph> graph)
      : m_graph(std::move(graph)) {}

  bool run() {
    return processBlocks(m_graph->block());
  }

 private:
  bool processBlocks(at::ArrayRef<torch::jit::Block*> blocks) {
    bool changed{false};
    // We are only interested in last block
    auto last_block_iter{blocks.rbegin()};
    if (last_block_iter != blocks.rend()) {
      changed |= processBlock(*last_block_iter);
    }
    return changed;
  }

  bool processBlock(torch::jit::Block* block) {
    bool changed{false};

    auto last_node_iter{block->nodes().rbegin()};

    if (last_node_iter != block->nodes().rend()) {
      torch::jit::Node* node{*last_node_iter};
      if (node->kind() != torch::jit::prim::TupleConstruct) {
        return changed;
      }

      block->removeAllOutputs();

      for (size_t input_idx = 0; input_idx < node->inputs().size();
           input_idx++) {
        block->insertOutput(input_idx, node->inputs()[input_idx]);
      }
      last_node_iter.destroyCurrent();
      changed |= true;
    }

    return changed;
  }

  std::shared_ptr<torch::jit::Graph> m_graph;
};

void SanitizeGraphInput(std::shared_ptr<torch::jit::Graph>& graph) {
  if (0 == graph->inputs().size()) {
    // No input to sanitize...
    return;
  }

  torch::jit::Value* first_graph_input{*graph->inputs().begin()};
  if (!first_graph_input->hasUses() &&
      "self" == first_graph_input->debugName()) {
    graph->eraseInput(0);
  }
  PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
}

void HandleTupleOnOutput(std::shared_ptr<torch::jit::Graph>& graph) {
  HandleTupleOnOutputPass pass{graph};
  pass.run();
  PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
}

} // namespace pass
} // namespace graph
} // namespace habana
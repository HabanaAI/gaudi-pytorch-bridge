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
    auto* return_node = block->return_node();
    if (return_node == nullptr || return_node->inputs().size() != 1)
      return false;

    auto* node = return_node->inputs()[0]->node();
    if (node == nullptr || node->kind() != torch::jit::prim::TupleConstruct)
      return false;

    block->removeAllOutputs();

    for (size_t input_idx = 0; input_idx < node->inputs().size(); input_idx++) {
      block->insertOutput(input_idx, node->inputs()[input_idx]);
    }

    node->destroy();
    return true;
  }

  std::shared_ptr<torch::jit::Graph> m_graph;
};

bool HandleTupleOnOutput(std::shared_ptr<torch::jit::Graph> graph) {
  PT_EAGER_TRACE;
  HandleTupleOnOutputPass pass{graph};
  bool changed{pass.run()};
  if (changed) {
    PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
  }
  return changed;
}

} // namespace pass
} // namespace graph
} // namespace habana
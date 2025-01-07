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
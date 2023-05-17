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
struct ConvertConvolutionPass {
  explicit ConvertConvolutionPass(std::shared_ptr<torch::jit::Graph> graph)
      : m_graph(std::move(graph)) {}

  bool run() {
    return processBlocks(m_graph->block());
  }

 private:
  bool processBlocks(at::ArrayRef<torch::jit::Block*> blocks) {
    bool changed{false};
    for (auto block : blocks) {
      changed |= processBlock(block);
    }
    return changed;
  }

  bool processBlock(torch::jit::Block* block) {
    bool changed{false};
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      changed |= processConvolution(it);
      changed |= processConvolutionBackward(it);
    }
    return changed;
  }

  bool processConvolution(torch::jit::graph_node_list_iterator& node) {
    static const auto conv_symbol{
        c10::Symbol::fromQualString("aten::convolution")};
    static const auto repl_symbol{
        c10::Symbol::fromQualString("aten::convolution_overrideable")};

    if (conv_symbol == node->kind()) {
      node->replaceWithNewSymbol(repl_symbol);
      node.destroyCurrent();
      return true;
    }
    return false;
  }

  bool processConvolutionBackward(torch::jit::graph_node_list_iterator& node) {
    static const auto conv_symbol{
        c10::Symbol::fromQualString("aten::convolution_backward")};
    static const auto repl_symbol{
        c10::Symbol::fromQualString("aten::convolution_backward_overrideable")};

    if (conv_symbol == node->kind()) {
      // Simple calling replaceWithNewSymbol will not suffice here.
      // We we have to omit one parameter to match schema for
      // aten::convolution_backward_overrideable
      // Note: This entire pass will not be necessary after aten::convolution
      // and aten::convolution_backward implementation will be provided in
      // backend
      torch::jit::WithInsertPoint insert_guard{*node};

      auto graph = node->owningGraph();
      auto replace_node = graph->insertNode(graph->create(repl_symbol, 0));
      at::ArrayRef<torch::jit::Value*> node_inputs{node->inputs()};
      static const size_t BIAS_SIZES_INPUT_IDX = 3;
      for (size_t input_idx = 0; input_idx < node_inputs.size(); input_idx++) {
        if (BIAS_SIZES_INPUT_IDX == input_idx) {
          continue;
        }
        replace_node->addInput(node_inputs[input_idx]);
      }
      for (torch::jit::Value* v : node->outputs()) {
        auto new_out = replace_node->addOutput()->copyMetadata(v);
        v->replaceAllUsesWith(new_out);
      }

      replace_node->copyMetadata(*node);
      replace_node->copyAttributes(**node);
      node.destroyCurrent();
      return true;
    }
    return false;
  }

  std::shared_ptr<torch::jit::Graph> m_graph;
  std::shared_ptr<std::vector<int>> m_inputs_to_permute;
};

void ConvertConvolutions(std::shared_ptr<torch::jit::Graph> graph) {
  PT_EAGER_TRACE;
  ConvertConvolutionPass pass{graph};
  bool changed{pass.run()};
  if (changed) {
    PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
  }
}

} // namespace pass
} // namespace graph
} // namespace habana

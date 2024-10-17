/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include "fuse_strided_views.h"

namespace habana_lazy {
void fuse_strided_views(std::shared_ptr<torch::jit::Graph>& graph) {
  torch::jit::graph_node_list graph_nodes = graph->nodes().reverse();
  using Node = torch::jit::Node;
  using namespace std::literals;
  std::vector<Node*> strided_view_nodes;
  for (auto* node : graph_nodes) {
    auto node_qual_str = std::string_view{node->kind().toQualString()};
    if (node_qual_str == "hpu::strided_view"sv) {
      strided_view_nodes.push_back(node);
    }
  }
  for (auto* node: strided_view_nodes){
    // Identify chain of strided views
    auto child_node_qual_str = std::string_view{node->input(0)->node()->kind().toQualString()};
    if (child_node_qual_str == "hpu::strided_view"sv) {
      Node* parent = node;
      Node* child = node->input(0)->node();
      // keep sizes and strides
      child->input(1)->replaceAllUsesWith(parent->input(1));
      child->input(2)->replaceAllUsesWith(parent->input(2));
      parent->output(0)->replaceAllUsesWith(child->output(0));
      parent->removeAllInputs();
      parent->destroy();
    }
  }
}
}; // namespace habana_lazy

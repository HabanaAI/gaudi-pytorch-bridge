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

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

#include "fuse_mm_transpose.h"

namespace habana_lazy {

void fuse_mm_transpose(std::shared_ptr<torch::jit::Graph>& graph) {
  torch::jit::graph_node_list graph_nodes = graph->nodes();
  using Node = torch::jit::Node;
  std::unordered_map<Node*, std::pair<Node*, Node*>> tmmt_nodes;
  for (auto* node : graph_nodes) {
    if (node->kind() == torch::jit::aten::t) {
      auto uses = node->output(0)->uses();
      for (auto u : uses) {
        auto mm_node = u.user;
        if (strcmp(mm_node->kind().toQualString(), "aten::mm") == 0) {
          auto single_use = mm_node->output(0)->uses();
          if (single_use.size() == 1) {
            auto tnode = single_use.at(0).user;
            if (strcmp(tnode->kind().toQualString(), "aten::t") == 0) {
              if (tmmt_nodes.find(mm_node) == tmmt_nodes.end())
                tmmt_nodes[mm_node] = {node, tnode};
            }
          }
        }
      }
    }
  }
  for (auto node : tmmt_nodes) {
    auto next_mm_node = node.first;
    auto first_t_node = node.second.first;
    auto last_t_node = node.second.second;
    torch::jit::WithInsertPoint insert_point(next_mm_node);
    auto op = c10::Symbol::fromQualString("hpu::mm_t");
    auto transpose_val = graph->insertConstant(at::IValue(true));
    auto no_transpose_val = graph->insertConstant(at::IValue(false));
    bool At_B_flag = next_mm_node->input(0) == first_t_node->output(0);
    bool A_Bt_flag = next_mm_node->input(1) == first_t_node->output(0);
    if (At_B_flag) {
      auto mm_t_tnode = graph->create(
          op,
          {next_mm_node->input(1),
           first_t_node->input(0),
           transpose_val,
           no_transpose_val},
          1);
      mm_t_tnode->output(0)->copyMetadata(last_t_node->output(0));
      mm_t_tnode->copyAttributes(*last_t_node);
      graph->insertNode(mm_t_tnode);
      last_t_node->output(0)->replaceAllUsesWith(mm_t_tnode->output(0));
      last_t_node->removeAllInputs();
      next_mm_node->destroy();
      last_t_node->destroy();
    }
    if (A_Bt_flag) {
      auto mm_t_tnode = graph->create(
          op,
          {first_t_node->input(0),
           next_mm_node->input(0),
           no_transpose_val,
           transpose_val},
          1);
      mm_t_tnode->output(0)->copyMetadata(last_t_node->output(0));
      mm_t_tnode->copyAttributes(*last_t_node);
      graph->insertNode(mm_t_tnode);
      last_t_node->output(0)->replaceAllUsesWith(mm_t_tnode->output(0));
      last_t_node->removeAllInputs();
      next_mm_node->destroy();
      last_t_node->destroy();
    }
  }
}
}; // namespace habana_lazy

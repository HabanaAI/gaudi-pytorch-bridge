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

using namespace torch::jit;

namespace habana_lazy {

using Graph = torch::jit::Graph;

/* */
bool is_inplace_or_views(torch::jit::Node* node) {
  bool is_inplace = false;
  auto node_str = node->kind().toQualString();
  size_t len = strlen(node_str);
  char endch = node_str[len - 1];

  if ((endch == '_') ||
      (strcmp(node->kind().toQualString(), "hpu::habana_d2d_memcpy_other") ==
       0)) {
    // TODO refine memcpy node check to add dst condition
    is_inplace = true;
  }

  return is_inplace;
}

void replace_views_with_reshapes(std::shared_ptr<Graph>& graph) {
  torch::jit::graph_node_list graph_nodes = graph->nodes();
  std::vector<Node*> as_strided_node_vec;

  // collect candidate as_strided nodes
  // TODO add other ops like view, slice etc once strided memcpy is available
  for (auto* node : graph_nodes) {
    if ((strcmp(node->kind().toQualString(), "hpu::as_strided_lazy_") == 0) ||
        (strcmp(node->kind().toQualString(), "hpu::as_strided_lazy_cl_") ==
         0)) {
      auto input_uses = node->input(0)->uses();
      auto output_uses = node->output(0)->uses();

      // check the 'can_replace' metadata to see if the node is a replacement
      // candidate
      bool is_replace_cand = toIValue(node->input(4)).value().toBool();

      // check if as_strided is as a graph output
      if (is_replace_cand == true) {
        for (auto& u : output_uses) {
          auto output_node = u.user;
          if (strcmp(output_node->kind().toQualString(), "prim::Return") == 0) {
            is_replace_cand = false;
          }
        }
      }

      if (is_replace_cand == true) {
        for (auto& u : node->input(0)->uses()) {
          auto input_node = u.user;
          if ((input_node != node) && is_inplace_or_views(input_node)) {
            is_replace_cand = false;
            break;
          }
        }
      }

      if (is_replace_cand == true) {
        for (auto& u : node->output(0)->uses()) {
          auto output_node = u.user;
          if ((output_node != node) && is_inplace_or_views(output_node)) {
            is_replace_cand = false;
            break;
          }
        }
      }

      if (is_replace_cand == true) {
        as_strided_node_vec.emplace_back(node);
      }
    } // if (strcmp(node->kind().toQualString(), "hpu::as_strided_lazy_") == 0)
  } // for (auto* node : graph_nodes)

  // 1. add a new reshape node
  // 2. remove the original as_strided node
  for (auto* node : as_strided_node_vec) {
    auto op = c10::Symbol::fromQualString("hpu::reshape");

    WithInsertPoint insert_point(node);
    auto new_reshape = graph->create(op, {node->input(0), node->input(1)}, 1);
    new_reshape->copyAttributes(*node);
    graph->insertNode(new_reshape);
    node->output(0)->replaceAllUsesWith(new_reshape->output(0));
    node->destroy();
  }
}

}; // namespace habana_lazy

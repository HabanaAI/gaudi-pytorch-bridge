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

#include "fuse_bn_relu_residual_add.h"

namespace habana_lazy {
/* Threshold backward takes relu's input as one of the input thereby preventing
the GC to fuse BN, residual add and relu nodes. Rewrite the graph to make
threshold backward take relu's output  instead. This works because of the
special property of threshold backward function bein invariant to Relu's
input/output*/

void fuse_bn_relu(std::shared_ptr<torch::jit::Graph>& graph) {
  torch::jit::graph_node_list graph_nodes = graph->nodes();

  std::vector<torch::jit::Node*> threshold_backward_node_vec;

  // collect threshold backward nodes
  for (auto* node : graph_nodes) {
    if (node->kind() == torch::jit::aten::threshold_backward) {
      threshold_backward_node_vec.emplace_back(node);
    }
  }

  // 1. add a new threshold backward node with relu's output as its input
  // 2. remove the original threshold backward node
  for (auto* node : threshold_backward_node_vec) {
    auto op = c10::Symbol::fromQualString("aten::threshold_backward");

    // determine the relu usage among all the uses
    for (auto u : node->input(1)->uses()) {
      auto u_node = u.user;

      if (strcmp(u_node->kind().toQualString(), "aten::relu") == 0) {
        torch::jit::WithInsertPoint insert_point(node);
        auto new_threshold_backward = graph->create(
            op, {node->input(0), u_node->output(0), node->input(2)}, 1);
        new_threshold_backward->output(0)->copyMetadata(node->output(0));
        new_threshold_backward->copyAttributes(*node);
        graph->insertNode(new_threshold_backward);
        node->output(0)->replaceAllUsesWith(new_threshold_backward->output(0));
        node->destroy();
        break;
      }
    }
  }
}
}; // namespace habana_lazy

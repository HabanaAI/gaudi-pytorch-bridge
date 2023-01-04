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
 *******************************************************************************
 */
#include "remove_redundant_memcpy.h"
#include "pytorch_helpers/util/jitgraph_utils.h"

using namespace torch::jit;
using namespace jitgraph_utils;
namespace habana_lazy {

using Graph = torch::jit::Graph;

static inline bool isRedundantMemcpyCandidate(Node* n) {
  return ((strcmp(n->kind().toQualString(), "hpu::habana_d2d_memcpy") == 0));
}

static bool isInList(const std::vector<Value*>& l, const Value* v) {
  return std::find(l.begin(), l.end(), v) != l.end();
}

static bool isGraphOutput(const std::shared_ptr<Graph>& graph, const Value* v) {
  auto outputs = graph->outputs().vec();
  if (isInList(outputs, v)) {
    return true;
  }

  for (auto& u : v->uses()) {
    auto n = u.user;
    if (n && isRedundantMemcpyCandidate(n)) {
      auto o = n->output(0);
      if (isGraphOutput(graph, o)) {
        return true;
      }
    }
  }

  return false;
}

// skipNode parameter:
// When considering inputs of node N, one of its 'uses' is node N.
// If we don't want to include itself in the condition check it has
// to be provided as skipNode.
// See use of this helper function in 'remove_redundant_memcpy' below.
static bool isInplaceOp(const Value* v, const Node* skipNode = nullptr) {
  for (auto& u : v->uses()) {
    auto n = u.user;
    if (n && (n != skipNode) && isInplace(n)) {
      return true;
    }
  }

  return false;
}

/* Objective of this pass is, if we have graph like this:
   A = SomeOp(B)
   C = Memcpy(A)
   D = SomeOtherOp(C)

   After optimization, it becomes
   A = SomeOp(B)
   D = SomeOtherOp(A)

   Conditions for this pass.
   1. Output shouldn't be graph output.
   2. Output shouldn't be input to inplace Op.
   3. Input to Memcpy shouldn't be graph output/input.
*/

void remove_redundant_memcpy(std::shared_ptr<Graph>& graph) {
  torch::jit::graph_node_list graph_nodes = graph->nodes();
  std::vector<Node*> redundant_memcpy_nodes;

  // collect redundant_memcpy_nodes
  for (Node* node : graph_nodes) {
    if (isRedundantMemcpyCandidate(node) &&
        !isInList(graph->inputs().vec(), node->input(0)) &&
        !isInList(graph->outputs().vec(), node->input(0)) &&
        node->output(0)->uses().size() == 1 &&
        !isInplaceOp(node->input(0), node) && !isInplaceOp(node->output(0))) {
      redundant_memcpy_nodes.emplace_back(node);
    }
  }

  for (auto node_itr = redundant_memcpy_nodes.rbegin();
       node_itr != redundant_memcpy_nodes.rend();
       ++node_itr) {
    Node* node = *node_itr;
    auto& uses = node->output(0)->uses();
    for (auto uses_itr = uses.rbegin(); uses_itr != uses.rend(); uses_itr++) {
      Node* output_node = (*uses_itr).user;
      output_node->replaceInputWith(node->output(0), node->input(0));
    }
    node->destroy();
  }
}
}; // namespace habana_lazy

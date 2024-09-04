/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#include "rem_dup_const_pass.h"

namespace habana_torch {
namespace jit {
bool RemoveDuplicateConstPass(habana_torch::jit::Graph& g) {
  bool graph_changed = false;
  std::list<Node*> const_nodes, nodes_to_remove;
  // todo: use bridge asserts https://jira.habana-labs.com/browse/SW-200787
  // GAUDI_JIT_DEBUG("Starting 'Remove duplicate const' pass.");
  for (Node* n : g.nodes()) {
    // Iterate only over const nodes
    if (n->kind() == prim::Constant) {
      auto val = toIValue(n->output()).value();
      auto node_it = find_if(
          const_nodes.begin(), const_nodes.end(), [&val](const Node* n) {
            auto n_val = toIValue(n->output()).value();
            return val == n_val;
          });
      if (node_it != const_nodes.end()) {
        // This const node already exists
        n->replaceAllUsesWith(*node_it);
        graph_changed = true;
        nodes_to_remove.push_back(n);
      } else {
        // Collect unique const nodes
        const_nodes.push_back(n);
      }
    }
  }
  // todo: use bridge infra https://jira.habana-labs.com/browse/SW-200787
  // GAUDI_JIT_DEBUG("Found ", nodes_to_remove.size(), " nodes to be removed.");
  // Remove duplicate nodes
  std::for_each(nodes_to_remove.begin(), nodes_to_remove.end(), [](Node* n) {
    TORCH_CHECK(!n->hasUses());
    // GAUDI_JIT_DEBUG("Removing node ", *n, ".");
    n->destroy();
  });
  // Move all the const nodes to the top
  if (!const_nodes.empty()) {
    Node* first_node = *(g.nodes().begin());
    std::for_each(const_nodes.rbegin(), const_nodes.rend(), [&](Node* n) {
      if (n != first_node) {
        n->moveBefore(first_node);
        graph_changed = true;
        first_node = n;
      }
    });
  }
  return graph_changed;
}
} // namespace jit
} // namespace habana_torch

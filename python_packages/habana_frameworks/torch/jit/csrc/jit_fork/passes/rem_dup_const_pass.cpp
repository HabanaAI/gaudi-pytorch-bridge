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
#include "rem_dup_const_pass.h"
#include "habana_helpers/logging.h"
namespace habana_torch {
namespace jit {
bool RemoveDuplicateConstPass(habana_torch::jit::Graph& g) {
  bool graph_changed = false;
  std::list<Node*> const_nodes, nodes_to_remove;
  PT_BRIDGE_DEBUG("Starting 'Remove duplicate const' pass.");
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
  PT_BRIDGE_DEBUG("Found ", nodes_to_remove.size(), " nodes to be removed.");
  // Remove duplicate nodes
  std::for_each(nodes_to_remove.begin(), nodes_to_remove.end(), [](Node* n) {
    TORCH_CHECK(!n->hasUses());
    PT_BRIDGE_DEBUG("Removing node ", *n, ".");
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

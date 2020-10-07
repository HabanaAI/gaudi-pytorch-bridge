/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "ir_utils.h"
#include "habana_helpers/logging.h"
#include "ir.h"

namespace habana_lazy {
namespace ir {
std::vector<NodePtr> Utils::ComputePostOrderNode(
    NodePtr p_node,
    EmissionMap* emap) {
  std::vector<NodePtr> post_order;
  std::vector<NodePtr> queue;
  queue.push_back(p_node);
  while (!queue.empty()) {
    p_node = queue.back();
    auto it = emap->find(p_node);
    if (it == emap->end()) {
      (*emap)[p_node] = kEmitting;

      for (auto& operand : p_node->GetInputs()) {
        auto oit = emap->find(operand.mp_node);
        if (oit == emap->end()) {
          queue.push_back(operand.mp_node);
        } else {
          // graph loop found at *operand.node
          HABANA_ASSERT(oit->second == kEmitting);
        }
      }
    } else if (it->second == kEmitting) {
      for (auto& operand : p_node->GetInputs()) {
        auto oit = emap->find(operand.mp_node);
        // check for graph loop at *operand.node
        HABANA_ASSERT(oit != emap->end() && oit->second == kEmitted);
      }
      (*emap)[p_node] = kEmitted;
      post_order.push_back(p_node);
      queue.pop_back();
    } else {
      HABANA_ASSERT(it->second == kEmitted);
      queue.pop_back();
    }
  }
  return post_order;
}

std::vector<NodePtr> Utils::ComputePostOrder(
    std::vector<NodePtr> p_nodes,
    EmissionMap* emap) {
  std::vector<NodePtr> post_order;
  for (auto p_node : p_nodes) {
    auto node_post_order = Utils::ComputePostOrderNode(p_node, emap);
    post_order.insert(
        post_order.end(), node_post_order.begin(), node_post_order.end());
  }
  return post_order;
}

} // namespace ir
} // namespace habana_lazy
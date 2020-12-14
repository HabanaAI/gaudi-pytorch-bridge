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

size_t Utils::StdHashCombine(uint64_t a, uint64_t b) {
  return a ^
      (b * 0x27d4eb2f165667c5 + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2));
}

/*
@brief - Computes post order traveral for a given output node
Computes input ir values asscoicated with the given output node
*/
void Utils::ComputePostOrderNode(
    NodePtr& p_node,
    EmissionMap* p_emap,
    NodePtrList& post_order,
    NodeSet& node_set,
    ValueList& inputs,
    size_t& post_order_nodes_hash) {
  PT_LAZY_TRACE;
  NodePtrList queue;
  queue.push_back(p_node);
  while (!queue.empty()) {
    p_node = queue.back();

    // check and update input value list
    auto operands = p_node->GetInputs();

    auto it = p_emap->find(p_node);
    if (it == p_emap->end()) {
      (*p_emap)[p_node] = kEmitting;

      for (auto& operand : operands) {
        std::string operand_node_kind = operand.mp_node->op().toQualString();
        auto oit = p_emap->find(operand.mp_node);

        if ("hpu::input" == operand_node_kind) {
          if (node_set.count(operand.mp_node) == 0) {
            node_set.insert(operand.mp_node);
            inputs.emplace_back(operand);
          }
        }

        if (oit == p_emap->end()) {
          queue.emplace_back(operand.mp_node);
        } else {
          // graph loop found at *operand.node
          // If the operand is in emap, it has to
          // be already emitted
          HABANA_ASSERT(oit->second == kEmitted);
        }
      }
    } else if (it->second == kEmitting) {
      for (auto& operand : operands) {
        auto oit = p_emap->find(operand.mp_node);
        // check for graph loop at *operand.node
        HABANA_ASSERT(oit != p_emap->end() && oit->second == kEmitted);
      }
      post_order_nodes_hash =
          torch::hash_combine(post_order_nodes_hash, p_node->get_hash());
      (*p_emap)[p_node] = kEmitted;
      post_order.emplace_back(p_node);
      queue.pop_back();

    } else {
      HABANA_ASSERT(it->second == kEmitted);
      queue.pop_back();
    }
  }
}

/*
@brief - Computes post order traveral across multiple output nodes
*/
void Utils::ComputePostOrder(
    NodePtrList& p_nodes,
    EmissionMap* emap,
    NodePtrList& post_order,
    ValueList& inputs,
    size_t& post_order_nodes_hash) {
  PT_LAZY_TRACE;
  NodeSet node_set;
  for (auto p_node : p_nodes) {
    Utils::ComputePostOrderNode(
        p_node, emap, post_order, node_set, inputs, post_order_nodes_hash);
  }
}

} // namespace ir
} // namespace habana_lazy

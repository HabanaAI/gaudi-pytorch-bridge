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
    NodePtrList& post_order) {
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
        auto oit = p_emap->find(operand.mp_node);
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
    NodePtrList& post_order) {
  for (auto p_node : p_nodes) {
    Utils::ComputePostOrderNode(p_node, emap, post_order);
  }
}

/*
@brief - Computes ValueList of inputs for the post ordered nodes. It would be
consumed in JIT IR creation
Algorithm:
1. For every 'non input' node pointer, fetch its input value pointers
2. If the NodePtr associated with the fetched value pointer is an input node,
update the ValueList.
3. In addition, mark the visited nodes to avoid duplicate updates
*/
void Utils::ComputePostOrderInputs(
    ValueList& input_val,
    NodePtrList& post_order) {
  auto sub_str = "hpu::input";
  for (auto p_node : post_order) {
    auto str = p_node->ToString();
    if (str.find(sub_str) == std::string::npos) {
      // check if its operands are inputs
      for (auto val : p_node->GetInputs()) {
        if (val.mp_node->IsVisited() == false) {
          val.mp_node->MarkVisited();
          auto input_str = val.mp_node->ToString();
          if (input_str.find(sub_str) != std::string::npos) {
            // operand is an input
            input_val.emplace_back(val);
          } // if (input_str.find(sub_str) != std::string::npos)
        } // if (val.mp_node->isVisited == false)
      } // for (auto val : p_node->GetInputs())
    } //  if (str.find(sub_str) == std::string::npos)
  } // for (auto p_node : post_order)
} // ComputePostOrderInputs()

} // namespace ir
} // namespace habana_lazy

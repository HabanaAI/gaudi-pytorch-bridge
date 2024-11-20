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
Computes input ir values associated with the given output node
*/
void Utils::ComputePostOrderNode(
    NodePtr& p_node,
    PostOrderData& po_data,
    NodeSet& node_set) {
  PT_LAZY_TRACE;
  NodePtrList queue;
  queue.push_back(p_node);
  while (!queue.empty()) {
    p_node = queue.back();

    // check and update input value list
    auto operands = p_node->GetInputs();

    auto it = po_data.emission_map.find(p_node);
    if (it == po_data.emission_map.end()) {
      po_data.emission_map[p_node] = EmitStatus::kEmitting;

      for (auto& operand : operands) {
        auto oit = po_data.emission_map.find(operand.mp_node);
        if (operand.mp_node->is_input()) {
          po_data.value_input_nodes_map[operand].emplace_back(p_node);
          if (node_set.count(operand.mp_node) == 0) {
            node_set.insert(operand.mp_node);
            po_data.inputs.emplace_back(operand);
          }
        }

        if (oit == po_data.emission_map.end()) {
          queue.emplace_back(operand.mp_node);
        } else {
          // graph loop found at *operand.node
          // If the operand is in emap, it has to
          // be already emitted
          HABANA_ASSERT(oit->second == EmitStatus::kEmitted);
        }
      }
    } else if (it->second == EmitStatus::kEmitting) {
      for (auto& operand : operands) {
        auto oit = po_data.emission_map.find(operand.mp_node);
        // check for graph loop at *operand.node
        HABANA_ASSERT(
            oit != po_data.emission_map.end() &&
            oit->second == EmitStatus::kEmitted);
      }
      po_data.post_order_nodes_hash =
          at::hash_combine(po_data.post_order_nodes_hash, p_node->get_hash());
      po_data.emission_map[p_node] = EmitStatus::kEmitted;
      p_node->set_post_order_pos(po_data.post_order.size());
      po_data.post_order.emplace_back(p_node);
      queue.pop_back();

    } else {
      HABANA_ASSERT(it->second == EmitStatus::kEmitted);
      queue.pop_back();
    }
  }
}

/*
@brief - Computes post order traveral across multiple output nodes
*/
void Utils::ComputePostOrder(NodePtrList& p_nodes, PostOrderData& po_data) {
  PT_LAZY_TRACE;
  NodeSet node_set;
  for (auto& p_node : p_nodes) {
    if (po_data.emission_map.count(p_node) == 0) {
      Utils::ComputePostOrderNode(p_node, po_data, node_set);
    }
  }
}

} // namespace ir
} // namespace habana_lazy

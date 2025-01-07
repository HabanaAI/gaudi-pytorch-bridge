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
#pragma once
#include <unordered_map>
#include <vector>
#include "ir.h"

namespace habana_lazy {
namespace ir {

// Tracks the emission status of the nodes during the post-order generation.
// It helps tracking loops within the computation graphs.
enum class EmitStatus {
  kNotEmitted,
  kEmitting,
  kEmitted,
};

using NodeSet = std::unordered_set<ir::NodePtr>;
using EmissionMap = std::unordered_map<NodePtr, EmitStatus>;
using ValueNodeListMap = std::unordered_map<
    ir::Value,
    std::vector<std::shared_ptr<Node>>,
    ir::ValueHash,
    ir::ValueEqual>;

struct PostOrderData {
  NodePtrList post_order;
  EmissionMap emission_map;
  ValueList inputs;
  ValueList outputs;
  ValueNodeListMap value_input_nodes_map;
  size_t post_order_nodes_hash = 0;
};

class Utils {
 public:
  static size_t StdHashCombine(uint64_t a, uint64_t b);

  // Computes the post order from the given node
  static void ComputePostOrderNode(
      NodePtr& p_node,
      PostOrderData& po_data,
      NodeSet& node_set);

  static void ComputePostOrder(NodePtrList& p_nodes, PostOrderData& po_data);
};

} // namespace ir
} // namespace habana_lazy

/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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

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

class Utils {
 public:
  // Tracks the emission status of the nodes during the post-order generation.
  // It helps tracking loops within the computation graphs.
  enum EmitStatus {
    kNotEmitted,
    kEmitting,
    kEmitted,
  };

  using NodeSet = std::unordered_set<ir::NodePtr>;
  using NodeValueMap = std::map<ir::NodePtr, ir::Value>;
  using EmissionMap = std::unordered_map<NodePtr, EmitStatus>;

  static size_t StdHashCombine(uint64_t a, uint64_t b);

  // Computes the post order from the given node
  static void ComputePostOrderNode(
      NodePtr& p_node,
      EmissionMap* emap,
      NodePtrList& post_order,
      NodeSet& node_set,
      ValueList& inputs,
      size_t& post_order_nodes_hash);

  static void ComputePostOrder(
      NodePtrList& p_nodes,
      EmissionMap* emap,
      NodePtrList& post_order,
      ValueList& inputs,
      size_t& post_order_nodes_hash);
};

} // namespace ir
} // namespace habana_lazy

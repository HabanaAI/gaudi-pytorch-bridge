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

  using EmissionMap = std::unordered_map<NodePtr, EmitStatus>;

  static size_t StdHashCombine(uint64_t a, uint64_t b);

  // Computes the post order from the given node
  static void ComputePostOrderNode(
      NodePtr& p_node,
      EmissionMap* emap,
      NodePtrList& post_order);

  static void ComputePostOrder(
      NodePtrList& p_nodes,
      EmissionMap* emap,
      NodePtrList& post_order);
  static void ComputePostOrderInputs(
      ValueList& input_val,
      NodePtrList& post_order);
};

} // namespace ir
} // namespace habana_lazy

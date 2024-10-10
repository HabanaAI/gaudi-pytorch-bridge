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

#include "backend/kernel/hpu_habana_launch_op_pt_sif_utils.h"

namespace habana::sif_utils {
void mapGraphInputsToInputsOnStack(
    const std::shared_ptr<torch::jit::Graph>& graph,
    const torch::jit::Stack& inputs,
    std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map) {
  auto stackIter = 0;
  for (const auto input : graph->inputs()) {
    val_to_ival_map[input] = inputs[stackIter++];
  }
}

c10::ScalarType getNodeScalarTypeFromInputs(
    const torch::jit::Node* node,
    const std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map) {
  // Default value is Float if no tensor is found
  auto node_type = c10::ScalarType::Float;
  for (auto input : node->inputs()) {
    if (auto inputIter = val_to_ival_map.find(input);
        inputIter != val_to_ival_map.end() and inputIter->second.isTensor()) {
      node_type = inputIter->second.toTensor().scalar_type();
      break;
    }
  }
  return node_type;
}

torch::jit::Stack createInputStackForNode(
    const torch::jit::Node* node,
    const std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map) {
  torch::jit::Stack stack;
  for (auto input : node->inputs()) {
    if (auto inputIter = val_to_ival_map.find(input);
        inputIter != val_to_ival_map.end()) {
      stack.push_back(inputIter->second);
    } else {
      HABANA_ASSERT(
          false,
          "Cannot proceed with unmapped input for ",
          node->kind().toQualString());
    }
  }
  return stack;
}
} // namespace habana::sif_utils

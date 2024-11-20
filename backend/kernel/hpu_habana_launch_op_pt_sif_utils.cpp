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

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
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/ir/ir.h>

namespace jitgraph_utils {

using Graph = torch::jit::Graph;
int64_t isInGraphInputs(const torch::jit::Value* value);
bool IsOutputToRestride(const torch::jit::Value* value);
torch::jit::Value* GetRestridedOutvalue(const torch::jit::Value* val);
torch::jit::Node* GetUnpackNodeFromTensorList(const torch::jit::Value* val);
bool isInGraphOutputs(const torch::jit::Node* node, size_t index);
bool isInGraphOutputs(const torch::jit::Node* node);
bool isInGraphOutputs(const torch::jit::Value* value);
bool isListNode(const torch::jit::Node* node);
int inplaceInputId(const torch::jit::Node* node);
bool isOutputCollective(const torch::jit::Node* node);

inline bool isInplace(const torch::jit::Node* node) {
  return inplaceInputId(node) >= 0;
}
c10::ArrayRef<torch::jit::Value*> getNodeOutputs(torch::jit::Node* node);
void visit_prim_node(
    const torch::jit::Node* node,
    std::unordered_map<const torch::jit::Value*, torch::jit::IValue>&
        val_to_ival_map);
} // namespace jitgraph_utils

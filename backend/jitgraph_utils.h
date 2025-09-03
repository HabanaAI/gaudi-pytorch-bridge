/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include "jit_fork/ir/ir.h"

namespace jitgraph_utils {

using Graph = habana_torch::jit::Graph;
int64_t isInGraphInputs(const habana_torch::jit::Value* value);
bool IsOutputToRestride(const habana_torch::jit::Value* value);
habana_torch::jit::Value* GetRestridedOutvalue(
    const habana_torch::jit::Value* val);
habana_torch::jit::Node* GetUnpackNodeFromTensorList(
    const habana_torch::jit::Value* val);
bool isInGraphOutputs(const habana_torch::jit::Node* node, size_t index);
bool isInGraphOutputs(const habana_torch::jit::Node* node);
bool isInGraphOutputs(const habana_torch::jit::Value* value);
bool isListNode(const habana_torch::jit::Node* node);
int inplaceInputId(const habana_torch::jit::Node* node);
bool isOutputCollective(const habana_torch::jit::Node* node);

inline bool isInplace(const habana_torch::jit::Node* node) {
  return inplaceInputId(node) >= 0;
}
c10::ArrayRef<habana_torch::jit::Value*> getNodeOutputs(
    habana_torch::jit::Node* node);
void visit_prim_node(
    const habana_torch::jit::Node* node,
    std::unordered_map<
        const habana_torch::jit::Value*,
        habana_torch::jit::IValue>& val_to_ival_map);
} // namespace jitgraph_utils

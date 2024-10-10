/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
#pragma once
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/ir/ir.h>

namespace jitgraph_utils {

using Graph = torch::jit::Graph;
int64_t isInGraphInputs(const torch::jit::Value* value);
bool IsOutputToRestride(const torch::jit::Value* value);
torch::jit::Value* GetRestridedOutvalue(const torch::jit::Value* val);
torch::jit::Value* GetPermuteOutvalue(const torch::jit::Value* val);
bool isPermuteInGraphOutputs(const torch::jit::Value* value);
bool IsOutputToPermute(const torch::jit::Value* value);
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

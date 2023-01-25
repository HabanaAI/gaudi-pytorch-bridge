/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

#include "jitgraph_utils.h"

namespace jitgraph_utils {

int64_t isInGraphInputs(const torch::jit::Value* value) {
  auto graph_ins = value->owningGraph()->inputs();
  auto it = std::find_if(
      graph_ins.cbegin(),
      graph_ins.cend(),
      [&](const torch::jit::Value* value_in) {
        return (value->unique() == value_in->unique());
      });

  if (it != graph_ins.cend()) {
    return (it - graph_ins.begin());
  }

  return -1;
}

torch::jit::Node* returnNodeUsesValue(
    const torch::jit::Value* value,
    std::list<std::string> Opslist) {
  auto uses = value->uses();
  for (auto u : uses) {
    auto restride_node = u.user;
    std::string str1 = restride_node->kind().toQualString();
    auto it = std::find(
        Opslist.begin(), Opslist.end(), restride_node->kind().toQualString());
    if (it != Opslist.end()) {
      return restride_node;
    }
  }
  return nullptr;
}

bool IsOutputToRestride(const torch::jit::Value* value) {
  return returnNodeUsesValue(value, {"hpu::restride_cl", "hpu::restride"})
      ? true
      : false;
}

torch::jit::Value* GetRestridedOutvalue(const torch::jit::Value* val) {
  auto restride_node =
      returnNodeUsesValue(val, {"hpu::restride_cl", "hpu::restride"});
  return restride_node ? restride_node->output(0) : nullptr;
}

torch::jit::Value* GetPermuteOutvalue(const torch::jit::Value* val) {
  auto restride_node = returnNodeUsesValue(val, {"hpu::permute"});
  return restride_node ? restride_node->output(0) : nullptr;
}

bool isPermuteInGraphOutputs(const torch::jit::Value* value) {
  // return if graph output is restrided node output
  if (IsOutputToPermute(value)) {
    auto value_permuted = GetPermuteOutvalue(value);
    TORCH_CHECK(nullptr != value_permuted, "Permuted value output is null");
    auto graph_outs = value->owningGraph()->outputs();
    for (auto value_out : graph_outs) {
      if (value_permuted->unique() == value_out->unique()) {
        return true;
      }
    }
  }
  return false;
}

bool IsOutputToPermute(const torch::jit::Value* value) {
  return returnNodeUsesValue(value, {"hpu::permute"}) ? true : false;
}

torch::jit::Node* GetUnpackNodeFromTensorList(const torch::jit::Value* val) {
  return returnNodeUsesValue(val, {"prim::ListUnpack"});
}

bool isInGraphOutputs(const torch::jit::Node* node, size_t index) {
  auto node_outs = node->outputs();
  TORCH_CHECK(index <= node_outs.size());

  return isInGraphOutputs(node_outs[index]);
}

bool isInGraphOutputs(const torch::jit::Value* value) {
  auto graph_outs = value->owningGraph()->outputs();
  for (auto value_out : graph_outs) {
    if (value->unique() == value_out->unique()) {
      return true;
    }
  }
  // return if graph output is restrided node output
  if (IsOutputToRestride(value)) {
    auto value_restrided = GetRestridedOutvalue(value);
    TORCH_CHECK(nullptr != value_restrided, "Restrided value output is null");
    auto graph_outs = value->owningGraph()->outputs();
    for (auto value_out : graph_outs) {
      if (value_restrided->unique() == value_out->unique()) {
        return true;
      }
    }
  }
  return false;
}

bool isListNode(const torch::jit::Node* node) {
  auto node_str = node->kind().toQualString();
  bool is_list_node = false;
  if ((strcmp(node_str, "prim::ListUnpack") == 0) ||
      (strcmp(node_str, "prim::ListConstruct") == 0)) {
    is_list_node = true;
  }
  return is_list_node;
}

bool isInplace(const torch::jit::Node* node) {
  auto node_name = node->kind().toQualString();
  bool is_inplace = false;
  size_t len = strlen(node_name);
  char endch = node_name[len - 1];
  if (endch == '_') {
    is_inplace = true;
  }
  return is_inplace;
}

} // namespace jitgraph_utils

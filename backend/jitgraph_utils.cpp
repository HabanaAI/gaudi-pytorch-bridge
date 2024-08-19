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

#include <cstring>

#include "backend/helpers/tensor_utils.h"
#include "backend/jitgraph_utils.h"
#include "habana_helpers/logging.h"

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

bool isOutputCollective(const torch::jit::Node* node) {
  for (auto node_outs : node->outputs()) {
    for (auto& u : node_outs->uses()) {
      if (habana_helpers::IsCollective(u.user->kind())) {
        return true;
      }
    }
  }
  return false;
}

bool isInGraphOutputs(const torch::jit::Node* node) {
  for (auto node_outs : node->outputs()) {
    if (isInGraphOutputs(node_outs)) {
      return true;
    }
  }
  return false;
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

int inplaceInputId(const torch::jit::Node* node) {
  auto node_name = node->kind().toQualString();
  size_t len = strlen(node_name);
  char endch = node_name[len - 1];
  char before_endch = (len > 1) ? node_name[len - 2] : ' ';
  int inputId = -1;
  // operators of the form op_ and __iop__ are inplace
  // but operators of the form op and __op__ are not:
  if ((endch == '_' && before_endch != '_') || strstr(node_name, "__i")) {
      inputId = 0;
  } else if (
      (strcmp(node_name, "hpu::habana_d2d_memcpy") == 0) ||
      (strcmp(node_name, "hpu::habana_d2d_memcpy_other") == 0)) {
    // Matching how MemCopyOperator::AllocateAndAddSynapseNode() calls
    // habana_helpers::duplicate_tensor_in_memory_section()
    inputId = (node->inputs().size() == 2) ? 1 : -1;
  }
  return inputId;
}

c10::ArrayRef<torch::jit::Value*> getNodeOutputs(torch::jit::Node* node) {
  auto node_outs = node->outputs();
  if (*node->output(0)->type() == *torch::jit::ListType::ofTensors() &&
      node->outputs().size() == 1) {
    auto unpack_node = GetUnpackNodeFromTensorList(node->output(0));
    HABANA_ASSERT(
        unpack_node != nullptr,
        "TensorList is not input to ListUnpack node. Node: ",
        node->kind().toQualString());
    node_outs = unpack_node->outputs();
  }
  return node_outs;
}

void visit_prim_node(
    const torch::jit::Node* node,
    std::unordered_map<const torch::jit::Value*, torch::jit::IValue>&
        val_to_ival_map) {
  if (torch::jit::prim::Constant == node->kind()) {
    for (const auto value : node->outputs()) {
      HABANA_ASSERT(val_to_ival_map.count(value) == 0);
      val_to_ival_map[value] = torch::jit::IValue(toIValue(value).value());
    }
  } else if (torch::jit::prim::ListConstruct == node->kind()) {
    auto node_outputs = node->outputs();
    HABANA_ASSERT(node_outputs.size() == 1);
    auto value{node_outputs[0]};
    const auto& node_ins = node->inputs();

    // Handle empty list
    if (node_ins.empty()) {
      val_to_ival_map[value] = torch::jit::IValue(c10::List<int64_t>());
      return;
    }

    auto in_ivalue_0 = val_to_ival_map[node_ins[0]];

    if (in_ivalue_0.isTensor()) {
      // Handle construction of list consisting tensor only
      std::vector<at::Tensor> tensorList;
      for (const auto input : node->inputs()) {
        HABANA_ASSERT(val_to_ival_map.count(input));
        auto input_ival = val_to_ival_map[input];
        HABANA_ASSERT(input_ival.isTensor());
        tensorList.emplace_back(input_ival.toTensor());
      }
      val_to_ival_map[value] = torch::jit::IValue(tensorList);
    } else if (in_ivalue_0.isInt()) {
      //  Handle construction of list consisting ints only
      c10::List<int64_t> intList;
      for (const auto& value_in : node_ins) {
        auto ivalue = val_to_ival_map[value_in];
        // Constructed list should be homogenous
        HABANA_ASSERT(ivalue.isInt());
        intList.emplace_back(ivalue.toInt());
      }
      val_to_ival_map[value] = torch::jit::IValue(intList);
    } else if (in_ivalue_0.isBool()) {
      //  Handle construction of list consisting bools only
      c10::List<bool> boolList;
      for (const auto& value_in : node_ins) {
        auto ivlaue = val_to_ival_map[value_in];
        // Constructed list should be homogenous
        HABANA_ASSERT(ivlaue.isBool());
        boolList.emplace_back(ivlaue.toBool());
      }
      val_to_ival_map[value] = torch::jit::IValue(boolList);
    }
  }
}

} // namespace jitgraph_utils

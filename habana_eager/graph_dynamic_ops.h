/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include <memory>
#include <string>
#include <vector>

#include <torch/csrc/jit/ir/ir.h>
#include "backend/synapse_helpers/layout_utils.h"
#include "habana_eager/graph_dynamic.h"

namespace habana {
namespace graph {

using GraphInputIndexMap = std::unordered_map<std::string, int64_t>;

class DynamicOp {
 public:
  virtual bool ReplaceWithDynamicHPUOp(
      torch::jit::Node* node,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      std::vector<at::Tensor>& in_tensors,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) = 0;

  static void UpdateDynamicInputs(
      std::vector<torch::jit::IValue*>& dtensor_list,
      std::vector<habana::graph::SymIntData>& scalar_list,
      std::vector<c10::IValue>& orig_stack);
  virtual ~DynamicOp() {}
};

using DynamicOpPtr = std::shared_ptr<DynamicOp>;
using DSOpRegisterFunc = std::function<DynamicOpPtr()>;

class RegisterDSOps {
 public:
  RegisterDSOps& add(const std::string guid, DSOpRegisterFunc func) {
    TORCH_CHECK(!dsOps_.count(guid), guid, " is already registered!");
    dsOps_.emplace(guid, func);
    return *this;
  }

  DynamicOpPtr get(const std::string& opname) {
    return dsOps_.count(opname) ? dsOps_[opname]() : nullptr;
  }

  RegisterDSOps() = default;
  RegisterDSOps(const RegisterDSOps&) = delete;
  RegisterDSOps& operator=(const RegisterDSOps&) = delete;

 private:
  std::unordered_map<std::string, DSOpRegisterFunc> dsOps_;
};

RegisterDSOps& DSOpsRegistry();

class ViewOperatorDS : public DynamicOp {
 public:
  ViewOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      std::vector<at::Tensor>& in_tensors,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      std::vector<torch::jit::IValue*>& dtensor_list,
      std::vector<habana::graph::SymIntData>& symint_list,
      std::vector<c10::IValue>& stack);
};

class RepeatOperatorDS : public DynamicOp {
 public:
  RepeatOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      std::vector<at::Tensor>& in_tensors,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      std::vector<torch::jit::IValue*>& dtensor_list,
      std::vector<habana::graph::SymIntData>& symint_list,
      std::vector<c10::IValue>& stack);
};

class TopkOperatorDS : public DynamicOp {
 public:
  TopkOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      std::vector<at::Tensor>& in_tensors,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      std::vector<torch::jit::IValue*>& dtensor_list,
      std::vector<habana::graph::SymIntData>& symint_list,
      std::vector<c10::IValue>& stack);
};

} // namespace graph
} // namespace habana

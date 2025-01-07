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

#include <memory>
#include <string>
#include <vector>

#include <torch/csrc/jit/ir/ir.h>
#include "backend/helpers/dynamic_shape_infer.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "habana_eager/graph_dynamic.h"
#include "habana_lazy/tensor_impl.h"

namespace habana {
namespace graph {

using GraphInputIndexMap = std::unordered_map<std::string, int64_t>;
void GetValueAndScalarIndexFromInput(
    torch::jit::Value* input,
    torch::jit::Stack& in_stack,
    GraphInputIndexMap& org_stack_index_map,
    int64_t& value,
    int64_t& index);
void GetValuesAndScalarIndexesFromListConst(
    torch::jit::Node* node,
    std::vector<int64_t>& values,
    std::vector<int64_t>& scalar_indexes);
void GetValuesAndScalarIndexesFromListConstruct(
    torch::jit::Node* node,
    torch::jit::Stack& in_stack,
    GraphInputIndexMap& org_stack_index_map,
    std::vector<int64_t>& values,
    std::vector<int64_t>& scalar_indexes);
std::string GetRangeInfoExprFromInput(
    torch::jit::Value* input,
    GraphInputIndexMap& org_stack_index_map,
    std::vector<habana_helpers::RangeInfo>* range_infos);
std::vector<std::string> GetRangeInfoExprFromListConstruct(
    torch::jit::Node* node,
    GraphInputIndexMap& org_stack_index_map,
    std::vector<habana_helpers::RangeInfo>* range_infos);
std::vector<std::string> GetRangeInfoExprFromListConst(
    torch::jit::Node* node,
    GraphInputIndexMap& org_stack_index_map,
    std::vector<habana_helpers::RangeInfo>* range_infos);
std::string GetExprFromString(std::vector<std::string> inputs);

torch::jit::Node* CreateAndInsertDynamicNodeToGraph(
    torch::jit::Graph* graph,
    torch::jit::Node* aten_view_node,
    const c10::Symbol& hpu_view_symbol,
    c10::ArrayRef<torch::jit::Value*> inputs,
    ValueIvalueMap& value_ivalue_map);

void UpdateShapeTensorSize(
    at::Tensor& dtensor,
    std::vector<int64_t>& stack_idxs,
    std::vector<c10::IValue>& orig_stack,
    LaunchDynamicShapes& launch_shapes);

int64_t UpdateDynamicTensorDSStack(
    torch::jit::IValue& iv_tensor,
    const std::vector<int64_t>& scalar_indexes,
    const std::vector<int64_t>& tensor_indexes,
    const std::vector<std::pair<int64_t, int64_t>>& mixed_indexes,
    std::shared_ptr<DynamicGraphMetaData> dmeta,
    const c10::SmallVector<int64_t, 8>& lookup_data = {});

int64_t CreateSTAndInsertToDSStack(
    const std::vector<int64_t>& st_size,
    const std::vector<int64_t>& scalar_indexes,
    const std::vector<int64_t>& tensor_indexes,
    const std::vector<std::pair<int64_t, int64_t>>& mixed_indexes,
    std::shared_ptr<DynamicGraphMetaData> dmeta,
    const c10::SmallVector<int64_t, 8>& lookup_data = {});

void UpdateH2DPatchingData(
    at::Tensor& dtensor,
    std::vector<int64_t>& data,
    LaunchDynamicShapes& launch_shapes);

template <typename T>
void UpdateH2DTensorData(at::Tensor& dtensor, std::vector<T>& data) {
  PT_EAGER_DEBUG("Input data for updating H2D tensor:", data);
  auto tmeta{get_tensor_extra_meta(dtensor)};
  tmeta->update_host_data(data.data(), data.size(), sizeof(T), true);
  PT_EAGER_DEBUG("Updated dynamic H2D tensor sizes:", dtensor.sizes());
}

template <typename T>
void SetH2DTensorHostData(
    at::Tensor& tensor,
    std::vector<T>& h2d_values,
    HostDataType dt_type,
    bool reverse_data) {
  auto tmeta{get_tensor_extra_meta(tensor)};
  std::vector<T> h2d_data(h2d_values);
  if (reverse_data) {
    std::reverse(h2d_data.begin(), h2d_data.end());
  }

  PT_EAGER_DEBUG(
      "Dynamic H2D tensor host data:",
      h2d_data,
      ", type:",
      dt_type,
      ", type size:",
      sizeof(T));
  tmeta->set_host_data(h2d_data.data(), h2d_data.size(), sizeof(T), dt_type);
  tmeta->set_H2D_data_for_bucketing();
}

template void SetH2DTensorHostData<int32_t>(
    at::Tensor&,
    std::vector<int32_t>&,
    HostDataType,
    bool);
template void SetH2DTensorHostData<uint32_t>(
    at::Tensor&,
    std::vector<uint32_t>&,
    HostDataType,
    bool);

template <typename T>
int64_t CreateH2DAndInsertToDSStack(
    std::vector<int64_t>& values,
    std::vector<int64_t>& scalar_indexes,
    HostDataType dt_type,
    std::shared_ptr<DynamicGraphMetaData> dmeta);

class DynamicOp {
 public:
  virtual bool ReplaceWithDynamicHPUOp(
      torch::jit::Node* node,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) = 0;

  virtual void ResolveNegativeSizes(
      [[maybe_unused]] torch::jit::Node* node,
      [[maybe_unused]] std::unordered_map<CValPtr, torch::jit::IValue>&
          value_ivalue_map,
      [[maybe_unused]] LaunchDynamicShapes& launch_shapes) {}

  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& scalar_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& orig_stack,
      LaunchDynamicShapes& launch_shapes);

  std::vector<at::Tensor> getInputTensors(
      const torch::jit::Node* node,
      ValueIvalueMap& value_ivalue_map) {
    std::vector<at::Tensor> in_tensors;
    for (const auto& input : node->inputs()) {
      PT_EAGER_DEBUG("Node input name = ", input->debugName());
      auto ivalue = value_ivalue_map[const_cast<torch::jit::Value*>(input)];
      HABANA_ASSERT(
          ivalue != nullptr,
          "Node = ",
          node->kind().toQualString(),
          ", input = ",
          input->debugName(),
          " not found in value_ivalue_map!!");
      if (ivalue->isTensor()) {
        in_tensors.push_back(ivalue->toTensor());
      }
    }
    return in_tensors;
  }

  std::vector<at::Tensor> getOutputTensers(
      const torch::jit::Node* node,
      ValueIvalueMap& value_ivalue_map) {
    std::vector<at::Tensor> out_tensors;
    for (const auto& output : node->outputs()) {
      PT_EAGER_DEBUG("Node output name = ", output->debugName());
      auto ivalue = value_ivalue_map[const_cast<torch::jit::Value*>(output)];
      HABANA_ASSERT(
          ivalue != nullptr,
          "Node = ",
          node->kind().toQualString(),
          ", output = ",
          output->debugName(),
          " not found in value_ivalue_map!!");
      if (ivalue->isTensor()) {
        out_tensors.push_back(ivalue->toTensor());
      }
    }
    return out_tensors;
  }
  std::map<int64_t, std::vector<int64_t>>* m_input_new_base_sizes = nullptr;
  std::vector<habana_helpers::RangeInfo>* m_range_infos = nullptr;
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

  // Used to pass list of registered DS ops to JIT pass in HPU
  // backend from HPU Eager Pass
  const std::vector<std::string> getRegisteredDSOpsList() const {
    std::vector<std::string> DSOpsList;
    for (const auto& op: dsOps_) {
      DSOpsList.push_back(op.first);
    }
    return DSOpsList;
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
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  void ResolveNegativeSizes(
      torch::jit::Node* node,
      std::unordered_map<CValPtr, torch::jit::IValue>& value_ivalue_map,
      LaunchDynamicShapes& launch_shapes) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class SelectScatterOperatorDS : public DynamicOp {
 public:
  SelectScatterOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
};

class SliceScatterOperatorDS : public DynamicOp {
 public:
  SliceScatterOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
};

class AsStridedScatterOperatorDS : public DynamicOp {
 public:
  AsStridedScatterOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class ArangeOperatorDS : public DynamicOp {
 public:
  ArangeOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class ConstantPad2dOperatorDS : public DynamicOp {
 public:
  ConstantPad2dOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class RepeatOperatorDS : public DynamicOp {
 public:
  RepeatOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class TopkOperatorDS : public DynamicOp {
 public:
  TopkOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class AsStridedOperatorDS : public DynamicOp {
 public:
  AsStridedOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& in_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class StridedInsertOperatorDS : public DynamicOp {
 public:
  StridedInsertOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& in_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class SliceOperatorDS : public DynamicOp {
 public:
  SliceOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& in_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class RandpermGeneratorOperatorDS : public DynamicOp {
 public:
  RandpermGeneratorOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class EmptyOpDS : public DynamicOp {
 public:
  EmptyOpDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class FullOpDS : public DynamicOp {
 public:
  FullOpDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class ExapndOperatorDS : public DynamicOp {
 public:
  ExapndOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& in_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class RandOperatorDS : public DynamicOp {
 public:
  RandOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class RandnOperatorDS : public DynamicOp {
 public:
  RandnOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

class RandintOperatorDS : public DynamicOp {
 public:
  RandintOperatorDS() : DynamicOp() {}
  bool ReplaceWithDynamicHPUOp(
      torch::jit::Node*,
      torch::jit::Stack& org_stack,
      GraphInputIndexMap& org_stack_index_map,
      ValueIvalueMap& value_ivalue_map,
      std::shared_ptr<DynamicGraphMetaData> m_dmeta) override;
  static void UpdateDynamicInputs(
      c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
      c10::SmallVectorImpl<habana::graph::SymIntData>& symint_list,
      c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
      c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&
          mixed_list,
      std::vector<c10::IValue>& stack,
      LaunchDynamicShapes& launch_shapes);
};

} // namespace graph
} // namespace habana

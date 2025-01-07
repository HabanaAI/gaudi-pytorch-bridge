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
#include "backend/habana_operator.h"
#include "backend/helpers/tensor_utils.h"

namespace habana {
class CompareOutOperator : public habana::HabanaOperator {
 public:
  CompareOutOperator(
      int device_id,
      c10::ScalarType scalarType,
      const std::string& guid)
      : HabanaOperator(guid), scalarType_(scalarType) {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

 protected:
  c10::ScalarType scalarType_;
};

class CompareOutWrapperOperator : public habana::HabanaOperator {
 public:
  CompareOutWrapperOperator(
      int device_id,
      c10::ScalarType scalarType,
      const std::string& guid)
      : HabanaOperator(guid), scalarType_(scalarType) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void SetPTOutputs(torch::jit::Stack& inputs) override;

 protected:
  c10::ScalarType scalarType_;
};

class CompareWrapperOperator : public CompareOutWrapperOperator {
 public:
  CompareWrapperOperator(
      int device_id,
      c10::ScalarType scalarType,
      const std::string& guid)
      : CompareOutWrapperOperator(device_id, scalarType, guid) {}

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void SetPTOutputs(torch::jit::Stack& inputs) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& arg1,
      const at::Tensor& arg2);
};

class GtOperator : public CompareWrapperOperator {
 public:
  GtOperator(int device_id, c10::ScalarType scalarType)
      : CompareWrapperOperator(
            device_id,
            scalarType,
            get_guid_with_precision("greater_fwd", scalarType)) {}
};

class EqOperator : public CompareWrapperOperator {
 public:
  EqOperator(int device_id, c10::ScalarType scalarType)
      : CompareWrapperOperator(
            device_id,
            scalarType,
            get_guid_with_precision("equal_fwd", scalarType)) {}
};

class LtOperator : public CompareWrapperOperator {
 public:
  LtOperator(int device_id, c10::ScalarType scalarType)
      : CompareWrapperOperator(
            device_id,
            scalarType,
            get_guid_with_precision("less_fwd", scalarType)) {}
};

} // namespace habana

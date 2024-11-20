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
#include "habana_kernels/tensor_shape_kernels.h"

namespace habana {

class BinaryInplaceOperator : public habana::HabanaOperator {
 public:
  BinaryInplaceOperator(
      int device_id,
      const std::string& guid,
      c10::ScalarType scalarType)
      : HabanaOperator(guid), scalarType_(scalarType) {
    this->CreateSynContext(device_id);
  }
  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

 protected:
  c10::ScalarType scalarType_;
};

class BinaryInplaceWrapperOperator : public habana::HabanaOperator {
 public:
  BinaryInplaceWrapperOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }
  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;

 protected:
  c10::ScalarType scalarType_;
};

class MulInplaceOperator : public BinaryInplaceWrapperOperator {
 public:
  // Mul op
  MulInplaceOperator(int device_id, c10::ScalarType scalarType)
      : BinaryInplaceWrapperOperator(
            device_id,
            get_guid_with_precision("mult", scalarType)) {
    scalarType_ = scalarType;
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::ANY});
  }
};
class BinaryInplaceOperatorWithAlpha : public BinaryInplaceOperator {
 public:
  BinaryInplaceOperatorWithAlpha(
      int device_id,
      const std::string& guid,
      c10::ScalarType scalarType)
      : BinaryInplaceOperator(device_id, guid, scalarType) {}
  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;
};

class BinaryInplaceWrapperOperatorWithAlpha : public habana::HabanaOperator {
 public:
  BinaryInplaceWrapperOperatorWithAlpha(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }
  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

 protected:
  c10::ScalarType scalarType_;
};

class AddInplaceOperator : public BinaryInplaceWrapperOperatorWithAlpha {
 public:
  AddInplaceOperator(int device_id, c10::ScalarType scalarType)
      : BinaryInplaceWrapperOperatorWithAlpha(
            device_id,
            get_guid_with_precision("add_fwd", scalarType)) {
    scalarType_ = scalarType;
  }
};
class AddcmulInplaceOperator : public habana::HabanaOperator {
 public:
  AddcmulInplaceOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("addcmul_fwd_") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};
} // namespace habana

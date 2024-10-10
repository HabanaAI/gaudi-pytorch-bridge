/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "backend/habana_operator.h"
#include "habana_kernels/tensor_shape_kernels.h"

namespace habana {

class BinaryOperator : public habana::HabanaOperator {
 public:
  BinaryOperator(
      int device_id,
      const std::string& guid,
      c10::ScalarType scalarType)
      : HabanaOperator(guid), scalarType_(scalarType) {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& arg1,
      const at::Tensor& arg2);

 protected:
  c10::ScalarType scalarType_;

  bool MaybeMultiplyWithBool(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaData& output_metadata);
};

class BinaryWrapperOperator : public habana::HabanaOperator {
 public:
  BinaryWrapperOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;
  void SetPTOutputs(torch::jit::Stack& inputs) override;

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

 protected:
  c10::ScalarType scalarType_;
};

class MulOperator : public BinaryWrapperOperator {
 public:
  // Mul op
  MulOperator(int device_id, c10::ScalarType scalarType)
      : BinaryWrapperOperator(
            device_id,
            get_guid_with_precision("mult", scalarType)) {
    scalarType_ = scalarType;
  }
};

class DivOperator : public BinaryWrapperOperator {
 public:
  DivOperator(int device_id, c10::ScalarType scalarType)
      : BinaryWrapperOperator(
            device_id,
            get_guid_with_precision("div_fwd", scalarType)) {
    scalarType_ = scalarType;
  }
};
class BinaryOperatorWithAlpha : public BinaryOperator {
 public:
  BinaryOperatorWithAlpha(
      int device_id,
      const std::string& guid,
      c10::ScalarType scalarType)
      : BinaryOperator(device_id, guid, scalarType) {}
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;
  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
};

class BinaryWrapperOperatorWithAlpha : public habana::HabanaOperator {
 public:
  BinaryWrapperOperatorWithAlpha(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
  void SetPTOutputs(torch::jit::Stack& inputs) override;

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

 protected:
  c10::ScalarType scalarType_;
};

class AddOperator : public BinaryWrapperOperatorWithAlpha {
 public:
  AddOperator(int device_id, c10::ScalarType scalarType)
      : BinaryWrapperOperatorWithAlpha(
            device_id,
            get_guid_with_precision("add_fwd", scalarType)) {
    scalarType_ = scalarType;
  }
};

class SubOperator : public BinaryWrapperOperatorWithAlpha {
 public:
  SubOperator(int device_id, c10::ScalarType scalarType)
      : BinaryWrapperOperatorWithAlpha(
            device_id,
            get_guid_with_precision("sub_fwd", scalarType)) {
    scalarType_ = scalarType;
  }
};

class RemainderOperator : public BinaryWrapperOperatorWithAlpha {
 public:
  RemainderOperator(int device_id, c10::ScalarType scalarType)
      : BinaryWrapperOperatorWithAlpha(
            device_id,
            get_guid_with_precision("rem_fwd", scalarType)) {
    scalarType_ = scalarType;
  }
};
} // namespace habana

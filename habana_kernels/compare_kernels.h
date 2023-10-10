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

/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include "habana_kernels/habana_operator.h"
namespace habana {

// Diag Operator
class DiagOutOperator : public habana::HabanaOperator {
 public:
  DiagOutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("diag") {
    this->CreateSynContext(device_id);
    static_cast<void>(scalarType);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
  void SetPTOutputs(torch::jit::Stack& inputs) override;
  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      int64_t& diagonal);
  at::Tensor AllocateOutputTensor(
      const at::Tensor& self,
      int64_t& diagonal,
      const OutputMetaData& output_metadata);
};

// Diag Operator
class DiagOperator : public DiagOutOperator {
 public:
  DiagOperator(int device_id, c10::ScalarType scalarType)
      : DiagOutOperator(device_id, scalarType) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

} // namespace habana
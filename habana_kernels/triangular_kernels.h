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

// Matrix Diagonal Operator
class MatrixDiagonalOperator : public habana::HabanaOperator {
 public:
  MatrixDiagonalOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "matrix_diagonal_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

// Diag Operator
class DiagOperator : public habana::HabanaOperator {
 public:
  DiagOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("diag") {
    this->CreateSynContext(device_id);
    static_cast<void>(scalarType);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
  void SetPTOutputs(torch::jit::Stack& inputs) override;

  at::Tensor AllocateOutputTensor(
      const at::Tensor& self,
      int64_t& diagonal,
      bool is_output_persistent);
  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      int64_t& diagonal);
};

// MatrixBandPart Operator
class MatrixBandPartOperator : public HabanaOperator {
 public:
  MatrixBandPartOperator(
      int device_id,
      c10::ScalarType scalarType,
      const std::string callerOp)
      : HabanaOperator(
            "matrix_band_part_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    this->callerOp_ = callerOp;
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

 private:
  std::string callerOp_;
};

// Triu Operator
class TriuOperator : public HabanaOperator {
 public:
  TriuOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(NULL_GUID) {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

// Tril Operator
class TrilOperator : public HabanaOperator {
 public:
  TrilOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(NULL_GUID) {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

} // namespace habana
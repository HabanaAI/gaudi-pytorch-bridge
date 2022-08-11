/******************************************************************************
 * Copyright (C) 2020-2022 Habana Labs, Ltd. an Intel Company
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

#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_operator.h"

namespace habana {

class MMOperator : public HabanaOperator {
 public:
  MMOperator(int device_id) : HabanaOperator("gemm") {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      at::Tensor self,
      at::Tensor other,
      bool self_transposed = false,
      bool other_transposed = false);
};

class AddmmOperator : public HabanaOperator {
 public:
  AddmmOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "gemm_add_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class BmmOutOperator : public HabanaOperator {
 public:
  BmmOutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("batch_gemm") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class BmmOperator : public BmmOutOperator {
 public:
  BmmOperator(int device_id, c10::ScalarType scalarType)
      : BmmOutOperator(device_id, scalarType) {
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      const at::Tensor& mat2);
};

//
// Mv Operator
class MvOperator : public HabanaOperator {
 public:
  MvOperator(int device_id) : HabanaOperator("mv") {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

//
// Dot Operator
class DotOperator : public HabanaOperator {
 public:
  DotOperator(int device_id) : HabanaOperator("dot") {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class MatMulOperator : public HabanaOperator {
 public:
  MatMulOperator(int device_id) : HabanaOperator("matmul") {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      const at::Tensor& mat2,
      bool other_transposed = false);
};

class MatmulBackwardOperator : public HabanaOperator {
 public:
  MatmulBackwardOperator(int device_id) : HabanaOperator("matmul_backward") {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

 private:
  synapse_helpers::tensor_or_ref MatBwTranspose(
      synapse_helpers::graph& graph,
      HabanaOperatorPtr Op,
      at::Tensor& mat,
      synapse_helpers::tensor_or_ref syn_input);

  std::tuple<synapse_helpers::tensor_or_ref, synapse_helpers::tensor_or_ref>
  MatBwSpecialFold(
      synapse_helpers::graph& graph,
      HabanaOperatorPtr Op,
      at::Tensor& mat1,
      at::Tensor& mat2,
      synapse_helpers::tensor_or_ref syn_input1,
      synapse_helpers::tensor_or_ref syn_input2);

  std::tuple<synapse_helpers::tensor_or_ref, synapse_helpers::tensor_or_ref>
  MatBwSize(
      synapse_helpers::graph& graph,
      HabanaOperatorPtr Op,
      at::Tensor& mat1,
      at::Tensor& mat2,
      at::IntArrayRef sizes,
      synapse_helpers::tensor_or_ref syn_input1,
      synapse_helpers::tensor_or_ref syn_input2,
      const OutputMetaData& output_metadata);

  synapse_helpers::tensor_or_ref MatBwReshape(
      synapse_helpers::graph& graph,
      at::Tensor& mat,
      std::vector<int64_t> sizes,
      synapse_helpers::tensor_or_ref syn_input);

  std::vector<HabanaOperatorPtr> ReshapeOpList;
};

class LinearForwardOperator : public HabanaOperator {
 public:
  LinearForwardOperator(int device_id) : HabanaOperator("linear_fwd") {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class LinearBackwardOperator : public HabanaOperator {
 public:
  LinearBackwardOperator(int device_id) : HabanaOperator("linear_bwd") {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

} // namespace habana

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
#include "backend/habana_operator.h"
namespace habana {

//
// EmbeddingBagSum Operator
class EmbeddingBagSumOperator : public HabanaOperator {
 public:
  EmbeddingBagSumOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            get_guid_with_precision("embedding_bag_sum_2d_fwd", scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY,
         LayoutFormat::ANY,
         LayoutFormat::ANY,
         LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

//
// EmbeddingBagSum Forward Operator
class EmbeddingBagSumForwardOperator : public HabanaOperator {
 public:
  EmbeddingBagSumForwardOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            get_guid_with_precision("embedding_bag_sum_2d_fwd", scalarType)),
        input_idx(0) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY,
         LayoutFormat::ANY,
         LayoutFormat::ANY,
         LayoutFormat::ANY,
         LayoutFormat::ANY,
         LayoutFormat::ANY,
         LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});

    for (auto idx = 0; idx < 4; idx++) {
      valid_input_idx.insert(idx);
    }
  }

  void AllocateSynapseInputs(
      synapse_helpers::graph& graph,
      const std::vector<at::Tensor>& inputs,
      bool is_persistent = false) override;

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  synapse_helpers::tensor& AllocateSynapseInput(
      synapse_helpers::graph& graph,
      const at::Tensor& input,
      bool is_persistent = false,
      synTensorType shape_tensor_type = DATA_TENSOR,
      void* host_ptr = nullptr,
      const std::string& idx = std::string()) override;

  synapse_helpers::tensor_or_ref& SetSynapseInput(
      synapse_helpers::tensor& tensor) override;

 private:
  int input_idx;
  std::set<int> valid_input_idx;
};

// Pad Operator
class PadOperator : public HabanaOperator {
 public:
  PadOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("pad_fwd", scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      c10::IntArrayRef pad);
  static std::vector<int64_t> compute_output_shape_ds(
      const at::Tensor& self,
      c10::IntArrayRef pad_before,
      c10::IntArrayRef pad_after);
};

// Pad (Host2Device Tensor) Operator
class PadOperatorHT : public PadOperator {
 public:
  PadOperatorHT(int device_id, c10::ScalarType scalarType)
      : PadOperator(device_id, scalarType) {}

  InferOutputMetaRetType InferOutputMeta(torch::jit::Stack& inputs) override;
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

//
// EmbeddingBagSum Backward out with kernel mode Operator
class EmbeddingBagSumBwdKernelModeOperator : public HabanaOperator {
 public:
  EmbeddingBagSumBwdKernelModeOperator(
      int device_id,
      c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision(
            "embedding_bag_sum_mid_lengths_2d_fwd",
            scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY,
         LayoutFormat::ANY,
         LayoutFormat::ANY,
         LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};
} // namespace habana

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

//
// EmbeddingBagSum Operator
class EmbeddingBagSumOperator : public HabanaOperator {
 public:
  EmbeddingBagSumOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "embedding_bag_sum_2d_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
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
            "embedding_bag_sum_2d_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
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
    input_idx = 0;
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
      void* host_ptr = nullptr) override;

  synapse_helpers::tensor_or_ref& SetSynapseInput(
      synapse_helpers::tensor& tensor) override;

 private:
  int input_idx;
  std::set<int> valid_input_idx;
};

//
// EmbeddingBagSum Backward Operator
class EmbeddingBagSumBackwardOperator : public HabanaOperator {
 public:
  EmbeddingBagSumBackwardOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "embedding_bag_sum_small_lengths_2d_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({
        LayoutFormat::ANY,
        LayoutFormat::ANY,
        LayoutFormat::ANY,
        LayoutFormat::ANY,
    });
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
    input_idx = 0;
    valid_input_idx.insert(1);
    valid_input_idx.insert(2);
    valid_input_idx.insert(3);
    valid_input_idx.insert(4);
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
      void* host_ptr = nullptr) override;

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
      : HabanaOperator(
            "pad_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
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

//
// Embedding Operator
class EmbeddingOperator : public HabanaOperator {
 public:
  EmbeddingOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "embedding_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

//
// Embedding Operator
class EmbeddingDenseBackwardOperator : public HabanaOperator {
 public:
  EmbeddingDenseBackwardOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "embedding_dense_bwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

 protected:
  std::string memcopy_guid;
};

//
// EmbeddingBagSum Backward out with kernel mode Operator
class EmbeddingBagSumBwdKernelModeOperator : public HabanaOperator {
 public:
  EmbeddingBagSumBwdKernelModeOperator(
      int device_id,
      c10::ScalarType scalarType)
      : HabanaOperator(
            "embedding_bag_sum_mid_lengths_2d_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
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

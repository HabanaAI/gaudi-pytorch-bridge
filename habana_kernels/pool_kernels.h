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
using namespace habana;

// Pool Operator
//
class MaxPool2dWithIndicesOperator : public HabanaOperator {
 public:
  MaxPool2dWithIndicesOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "maxpool_2d_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::NHWC,
                                           LayoutFormat::ANY,
                                           LayoutFormat::ANY,
                                           LayoutFormat::ANY,
                                           LayoutFormat::ANY,
                                           LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign(
        {LayoutFormat::NHWC, LayoutFormat::NHWC});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      std::vector<bool> is_output_persistent) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

class MaxPool2dOperator : public MaxPool2dWithIndicesOperator {
 public:
  MaxPool2dOperator(int device_id, c10::ScalarType scalarType)
      : MaxPool2dWithIndicesOperator(device_id, scalarType) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::NHWC});
    kernel_meta_data_.output_layout.assign(
        {LayoutFormat::NHWC, LayoutFormat::NHWC});
    p_context_->excluded_output_indices_ = {0};
  }
};

class MaxPool2dWithIndicesBackwardOutOperator : public HabanaOperator {
 public:
  MaxPool2dWithIndicesBackwardOutOperator(
      int device_id,
      c10::ScalarType scalarType)
      : HabanaOperator(
            "maxpool_2d_bwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);

    kernel_meta_data_.input_layout.assign({LayoutFormat::NHWC,
                                           LayoutFormat::NHWC,
                                           LayoutFormat::NHWC,
                                           LayoutFormat::NHWC});
    kernel_meta_data_.output_layout.assign({LayoutFormat::NHWC});
    kernel_meta_data_.tpc_input_order = {0, 2};
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

//
// MaxPool2dWithIndicesBackward Operator
class MaxPool2dWithIndicesBackwardOperator
    : public MaxPool2dWithIndicesBackwardOutOperator {
 public:
  MaxPool2dWithIndicesBackwardOperator(
      int device_id,
      c10::ScalarType scalarType)
      : MaxPool2dWithIndicesBackwardOutOperator(device_id, scalarType) {
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::NHWC, LayoutFormat::NHWC, LayoutFormat::NHWC});
    kernel_meta_data_.output_layout.assign({LayoutFormat::NHWC});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

class AvgPool2dOperator : public HabanaOperator {
 public:
  AvgPool2dOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "avg_pool_2d_fwd_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::NHWC});
    kernel_meta_data_.output_layout.assign({LayoutFormat::NHWC});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

class AvgPool2dBackwardOutOperator : public HabanaOperator {
 public:
  AvgPool2dBackwardOutOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "avg_pool_2d_bwd_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::NHWC, LayoutFormat::NHWC, LayoutFormat::NHWC});
    kernel_meta_data_.output_layout.assign({LayoutFormat::NHWC});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

class AvgPool2dBackwardOperator : public AvgPool2dBackwardOutOperator {
 public:
  AvgPool2dBackwardOperator(int device_id, c10::ScalarType scalar_type)
      : AvgPool2dBackwardOutOperator(device_id, scalar_type) {
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::NHWC, LayoutFormat::NHWC});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};
class PoolHelper {
 public:
  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& input,
      const at::IntArrayRef kernel_size,
      const at::IntArrayRef stride,
      const at::IntArrayRef padding,
      const at::IntArrayRef dilation,
      bool ceil_mode,
      bool is_input_nhwc);
};
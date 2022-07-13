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
    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;

  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;
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
    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN}); // shape tensor
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    kernel_meta_data_.tpc_input_order = {0, 2};
  }
  virtual habana::OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
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
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
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
    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
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
    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN}); // shape tensor
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    kernel_meta_data_.tpc_input_order = {0};
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
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
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
};

class AdaptiveAvgPool2dOperator : public HabanaOperator {
 public:
  AdaptiveAvgPool2dOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "adaptive_avg_pool_2d_fwd_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::NHWC});
    kernel_meta_data_.output_layout.assign({LayoutFormat::NHWC});
    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
};

class AdaptiveAvgPool2dBackwardOperator : public HabanaOperator {
 public:
  AdaptiveAvgPool2dBackwardOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "adaptive_avg_pool_2d_bwd_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::NHWC, LayoutFormat::NHWC});
    kernel_meta_data_.output_layout.assign({LayoutFormat::NHWC});
    kernel_meta_data_.tpc_input_order = {0};
    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
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

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& input,
      const at::IntArrayRef output_size,
      bool is_input_nhwc);

  static std::vector<int64_t> compute_output_shape_synapse(
      const at::Tensor& input,
      const at::IntArrayRef kernel_size,
      const at::IntArrayRef stride,
      const at::IntArrayRef padding,
      const at::IntArrayRef dilation,
      bool ceil_mode);

  static std::vector<int64_t> compute_output_shape_synapse(
      const at::Tensor& input,
      const at::IntArrayRef output_size);
};
} // namespace habana

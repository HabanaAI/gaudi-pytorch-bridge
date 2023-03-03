/*******************************************************************************
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
#include "backend/habana_operator.h"
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
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::NHWC,
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

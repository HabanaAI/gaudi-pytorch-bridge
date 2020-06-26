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
      bool is_output_persistent = false) override;

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

class MaxPool2dWithIndicesBackwardOperator : public HabanaOperator {
 public:
  MaxPool2dWithIndicesBackwardOperator(
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
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

class AvgPool2dOperator : public HabanaOperator {
 public:
  AvgPool2dOperator(int device_id, std::string guid) : HabanaOperator(guid) {
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
  AvgPool2dBackwardOutOperator(int device_id, std::string guid)
      : HabanaOperator(guid) {
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
  AvgPool2dBackwardOperator(int device_id, std::string guid)
      : AvgPool2dBackwardOutOperator(device_id, guid) {
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::NHWC, LayoutFormat::NHWC});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

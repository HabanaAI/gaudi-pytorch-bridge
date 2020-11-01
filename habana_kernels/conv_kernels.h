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
class ConvOperator : public habana::HabanaOperator {
 public:
  ConvOperator(int device_id, c10::ScalarType scalarType);
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);

  virtual void SetPTOutputs(torch::jit::Stack& inputs);

  static std::vector<int64_t> compute_output_shape(
    std::vector<int64_t> shape_in,
    std::vector<int64_t> shape_wt,
    std::vector<int64_t> pad,
    std::vector<int64_t> stride,
    const bool ceil_mode,
    c10::MemoryFormat memory_format);

 private:
  c10::ScalarType scalarType_;
  std::vector<synapse_helpers::tensor_or_ref> tensors_;
};

//
// For suporting conv2d operation from Graph mode
class Conv2dOperator : public ConvOperator {
 public:
  Conv2dOperator(int device_id, c10::ScalarType scalarType)
      : ConvOperator(device_id, scalarType) {}
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent) {
    TORCH_CHECK(
        inputs.size() == 7, "Conv2d Operation expects 7 arguments as input")
    bool transposed = false;
    int64_t out_padding[] = {0, 0, 0, 0};
    c10::IntArrayRef output_padding = out_padding;
    inputs.insert(inputs.begin() + 6, c10::IValue(transposed));
    inputs.insert(inputs.begin() + 7, c10::IValue(output_padding));
    ConvOperator::AllocateAndAddSynapseNode(
        graph, inputs, is_output_persistent);
  }
};

class ConvBackwardOperator : public HabanaOperator {
 public:
  ConvBackwardOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("convolution_bwd") {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::NHWC, LayoutFormat::NHWC, LayoutFormat::HWCK});
    kernel_meta_data_.output_layout.assign(
        {LayoutFormat::NHWC, LayoutFormat::HWCK, LayoutFormat::ANY});
  }

  virtual void SetPTOutputs(torch::jit::Stack& inputs);
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      std::vector<bool> is_output_persistent) override;
};

class ConvInputDifferentiationOperator : public HabanaOperator {
 public:
  ConvInputDifferentiationOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::NHWC, LayoutFormat::NHWC, LayoutFormat::HWCK});
    kernel_meta_data_.output_layout.assign({LayoutFormat::NHWC});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

class ConvWeightDifferentiationOperator : public HabanaOperator {
 public:
  ConvWeightDifferentiationOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::NHWC, LayoutFormat::NHWC, LayoutFormat::HWCK});
    kernel_meta_data_.output_layout.assign({LayoutFormat::HWCK});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

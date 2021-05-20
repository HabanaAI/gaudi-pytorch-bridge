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

/**
 * @brief Class implementing Pytorch "convolution_backward_overrideable"
 * operator for Habana device
 **/
class ConvBackwardOperator : public HabanaOperator {
 public:
  ConvBackwardOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("convolution_overrideable_bwd") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
    // Note that when this operator class is used for Conv2d_bwd operation (on
    // 4d input & weight tensors), then weights need to be in HWCK layout. But
    // when this operator class is used for Conv_transpose2d_bwd operation
    // (transpose = true), then weights need to be in HWKC layout. Since Pytorch
    // starts with CKHW layout for conv_transpose2d (unlike conv2d where its
    // KCHW), therefore all permutations done on weights can be handled in
    // bridge same way as those done for regular conv2d.
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::NHWC, LayoutFormat::NHWC, LayoutFormat::HWCK});
    kernel_meta_data_.output_layout.assign(
        {LayoutFormat::NHWC, LayoutFormat::HWCK, LayoutFormat::NHWC});
  }

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      std::vector<bool> is_output_persistent) override;

 private:
  void ComputeBiasGrad(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      std::vector<bool> is_output_persistent,
      bool mask_grad_in);
};

/**
 * @brief Internal class implementing Syanpse "dedx"
 * operator. Class objects to this should be invoked only from
 * other convolution related classes.
 **/
class ConvInputDifferentiationOperator : public HabanaOperator {
 public:
  ConvInputDifferentiationOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

/**
 * @brief Internal class implementing Syanpse "dedw"
 * operator. Class objects to this should be invoked only from
 * other convolution related classes.
 **/
class ConvWeightDifferentiationOperator : public HabanaOperator {
 public:
  ConvWeightDifferentiationOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};
} // namespace habana

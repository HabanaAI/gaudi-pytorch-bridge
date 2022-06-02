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

  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

 private:
  void ComputeBiasGrad(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata,
      bool mask_grad_in);

  void ComputeBiasGrad3d(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata,
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
    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::SRCK,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  }

  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

/**
 * @brief Internal class implementing Syanpse "dedx3d"
 * operator. Class objects to this should be invoked only from
 * other convolution related classes.
 **/
class Conv3dInputDifferentiationOperator : public HabanaOperator {
 public:
  Conv3dInputDifferentiationOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
         synapse_helpers::layouts::SynapseLayoutFormat::SRQCK,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDCN});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN});
  }

  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
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
    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::SRCK});
  }

  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

/**
 * @brief Internal class implementing Syanpse "dedw3d"
 * operator. Class objects to this should be invoked only from
 * other convolution related classes.
 **/
class Conv3dWeightDifferentiationOperator : public HabanaOperator {
 public:
  Conv3dWeightDifferentiationOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDCN});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::SRQCK});
  }

  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};
} // namespace habana

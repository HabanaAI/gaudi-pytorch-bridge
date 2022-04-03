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

// index of hieght/width/depth in pad/stride/dial tensor
#define CONV2D_KERNEL_HIEGHT_ATTRIBUTE_IDX 0
#define CONV2D_KERNEL_WIDTH_ATTRIBUTE_IDX 1
#define CONV3D_KERNEL_DEPTH_ATTRIBUTE_IDX 0
#define CONV3D_KERNEL_HIEGHT_ATTRIBUTE_IDX 1
#define CONV3D_KERNEL_WIDTH_ATTRIBUTE_IDX 2
namespace habana {
/**
 * @brief Class implementing Pytorch "convolution_overrideable"
 * operator for Habana device
 **/
class ConvOperator : public habana::HabanaOperator {
 public:
  ConvOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("convolution_overrideable") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
    // Note that when this operator class is used for Conv2d operation (on 4d
    // input & weight tensors), then weights need to be in HWCK layout. But when
    // this operator class is used for Conv_transpose2d operation (transpose =
    // true), then weights need to be in HWKC layout. Since Pytorch starts with
    // CKHW layout for conv_transpose2d (unlike conv2d where its KCHW),
    // therefore all permutations done on weights can be handled in bridge same
    // way as those done for regular conv2d.
    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NHWC,
         habana::LayoutFormat::HWCK,
         habana::LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NHWC});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;

  static std::vector<int64_t> compute_output_shape(
      std::vector<int64_t> shape_in,
      std::vector<int64_t> shape_wt,
      std::vector<int64_t> pad,
      std::vector<int64_t> stride,
      std::vector<int64_t> dilation,
      const bool ceil_mode,
      const bool transposed,
      c10::MemoryFormat memory_format,
      const bool is_conv_3d = false,
      const bool is_weight_hwck = true);

  static std::vector<int64_t> compute_output_shape(
      std::vector<int64_t> shape_in,
      std::vector<int64_t> shape_wt,
      std::vector<int64_t> pad,
      std::vector<int64_t> stride,
      std::vector<int64_t> dilation,
      const bool ceil_mode,
      const bool transposed);

 private:
  static std::vector<int64_t> compute_output_shape_2d(
      std::vector<int64_t> shape_in,
      std::vector<int64_t> shape_wt,
      std::vector<int64_t> pad,
      std::vector<int64_t> stride,
      std::vector<int64_t> dilation,
      const bool ceil_mode,
      const bool transposed);

  static std::vector<int64_t> compute_output_shape_3d(
      std::vector<int64_t> shape_in,
      std::vector<int64_t> shape_wt,
      std::vector<int64_t> pad,
      std::vector<int64_t> stride,
      std::vector<int64_t> dilation,
      const bool ceil_mode,
      const bool transposed);

  static int64_t compute_output_single_dim(
      std::vector<int64_t> shape_in,
      std::vector<int64_t> shape_wt,
      std::vector<int64_t> padding,
      std::vector<int64_t> strides,
      std::vector<int64_t> dilation,
      unsigned input_idx,
      unsigned kernel_idx,
      unsigned attributes_idx,
      bool transposed);
};

/**
 * @brief Internal class implementing Syanpse spatial_convolution
 * operator. Class objects to this should be invoked only from
 * other convolution related classes.
 **/
class SpatialConvOperator : public habana::HabanaOperator {
 public:
  SpatialConvOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("spatial_convolution") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);

    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::SRCK,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

/**
 * @brief Internal class implementing Syanpse spatial_convolution3d
 * operator. Class objects to this should be invoked only from
 * other convolution related classes.
 **/
class SpatialConv3DOperator : public habana::HabanaOperator {
 public:
  SpatialConv3DOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("spatial_convolution3d") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);

    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
         synapse_helpers::layouts::SynapseLayoutFormat::SRQCK,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};
} // namespace habana

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
      bool is_output_persistent = false);

  virtual void SetPTOutputs(torch::jit::Stack& inputs);

  static std::vector<int64_t> compute_output_shape(
      std::vector<int64_t> shape_in,
      std::vector<int64_t> shape_wt,
      std::vector<int64_t> pad,
      std::vector<int64_t> stride,
      const bool ceil_mode,
      const bool transposed,
      c10::MemoryFormat memory_format);
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
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);
};
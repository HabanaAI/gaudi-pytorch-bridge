/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "backend/helpers/tensor_utils.h"

namespace habana {

class UpsampleOperator : public HabanaOperator {
 public:
  UpsampleOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::NHWC});
    kernel_meta_data_.output_layout.assign({LayoutFormat::NHWC});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata);
  virtual void SetPTOutputs(torch::jit::Stack& inputs);
  static std::vector<int64_t> compute_output_shape(
      std::vector<int64_t> shape_in,
      OptionalIntArrayRef output_size,
      c10::optional<at::ArrayRef<double>> scales,
      c10::MemoryFormat memory_format);
  static std::vector<int64_t> compute_output_shape(
      std::vector<int64_t> shape_in,
      OptionalIntArrayRef output_size,
      c10::optional<at::ArrayRef<double>> scales);
  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;
};

class UpsampleNearest2dOperator : public UpsampleOperator {
 public:
  UpsampleNearest2dOperator(int device_id, c10::ScalarType scalarType)
      : UpsampleOperator(
            device_id,
            "resize_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  }
};

} // namespace habana

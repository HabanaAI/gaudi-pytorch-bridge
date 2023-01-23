/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include "backend/habana_operator.h"

namespace habana {
struct RepeatOperator : public HabanaOperator {
  RepeatOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "tile_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      at::IntArrayRef repeats);

  static std::vector<int64_t> compute_reshape_output(
      const at::Tensor& self,
      at::IntArrayRef repeats);
};

struct RepeatOperatorHT : public RepeatOperator {
  RepeatOperatorHT(int device_id, c10::ScalarType scalarType)
      : RepeatOperator(device_id, scalarType) {}

  OutputShapeInfRetType ComputeOutputShape(torch::jit::Stack& inputs);
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class RepeatInlvOperator : public HabanaOperator {
 public:
  RepeatInlvOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "repeat_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    CreateSynContext(device_id);
  }
  OutputShapeInfRetType ComputeOutputShape(torch::jit::Stack& inputs);
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& input,
      int64_t dim,
      int64_t out_size);
};
} // namespace habana

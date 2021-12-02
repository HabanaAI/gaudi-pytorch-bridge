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
#include "habana_operator.h"

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

  void AllocateAndAddSynapseNode(
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

class RepeatInlvOperator : public HabanaOperator {
 public:
  RepeatInlvOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "repeat_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    CreateSynContext(device_id);
  }

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

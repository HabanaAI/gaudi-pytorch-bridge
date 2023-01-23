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
#include "backend/habana_operator.h"

namespace habana {

class ThresholdBackwardOperator : public HabanaOperator {
 public:
  ThresholdBackwardOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "relu_bwd_" + habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata);
  virtual habana::OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;
};
} // namespace habana

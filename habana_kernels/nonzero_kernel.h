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

// NonZero Operator
class NonZeroOperator : public HabanaOperator {
 public:
  NonZeroOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("non_zero_fwd", scalarType)) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void SetPTOutputs(torch::jit::Stack& inputs) override;

  static std::vector<int64_t> compute_output_shape(const at::Tensor& input);
  static float round_dims(const at::Tensor& input_tensor, int group_size);

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  static std::vector<int64_t> compute_output_st_shape(
      const at::Tensor& input_tensor);
};
} // namespace habana

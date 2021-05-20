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
#include "habana_kernels/habana_operator.h"
namespace habana {

class WhereOperator : public HabanaOperator {
 public:
  WhereOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "where_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& condition,
      const at::Tensor& self,
      const at::Tensor& other);
  void SetPTOutputs(torch::jit::Stack& inputs) override;
};

} // namespace habana

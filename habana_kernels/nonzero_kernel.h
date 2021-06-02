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

// NonZero Operator
class NonZeroOperator : public HabanaOperator {
 public:
  NonZeroOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "non_zero_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      std::vector<bool> is_output_persistent) override;

  void SetPTOutputs(torch::jit::Stack& inputs) override;
};
} // namespace habana

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
#include <torch/script.h>
#include "habana_kernels/habana_operator.h"
using namespace torch;
using namespace habana;

class OptimizerLambPhase1Operator : public HabanaOperator {
 public:
  OptimizerLambPhase1Operator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "optimizer_lamb_ph1_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent) override;
};

class OptimizerLambPhase2Operator : public HabanaOperator {
 public:
  OptimizerLambPhase2Operator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "optimizer_lamb_ph2_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent) override;
};

class OptNormFusedNormOperator : public HabanaOperator {
 public:
  OptNormFusedNormOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "opt_lamb_fused_norm_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent) override;
};
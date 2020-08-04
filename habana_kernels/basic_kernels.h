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

//
// ToDtype Operator
class ToDtypeOperator : public habana::HabanaOperator {
 public:
  ToDtypeOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("to_dtype") {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  // virtual void SetPTOutput(torch::jit::Stack& inputs) override;
};

//
// MemCopy Operator
class MemCopyOperator : public habana::HabanaOperator {
 public:
  MemCopyOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("memcpy") {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

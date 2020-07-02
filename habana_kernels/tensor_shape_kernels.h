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
// Permute Operator
class PermuteOperator : public habana::HabanaOperator {
 public:
  PermuteOperator(int device_id, c10::ScalarType scalarType);
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

//
// Reshape Operator
class ReshapeOperator : public habana::HabanaOperator {
 public:
  ReshapeOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("reshape") {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

//
// Transpose Operator
class TransposeOperator : public habana::HabanaOperator {
 public:
  TransposeOperator(int device_id, c10::ScalarType scalarType);
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

//
// aten::t operator implementattion
class TOperator : public TransposeOperator {
 public:
  TOperator(int device_id, c10::ScalarType scalarType)
      : TransposeOperator(device_id, scalarType) {}

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent) override {
    TORCH_CHECK(
        inputs.size() == 1, "aten::t Operation expects 1 arguments as input")
    inputs.insert(inputs.begin() + 1, c10::IValue(0));
    inputs.insert(inputs.begin() + 2, c10::IValue(1));
    TransposeOperator::AllocateAndAddSynapseNode(
        graph, inputs, is_output_persistent);
  }
};

// Broadcast Operator
class BroadcastOperator : public habana::HabanaOperator {
 public:
  BroadcastOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("broadcast") {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};
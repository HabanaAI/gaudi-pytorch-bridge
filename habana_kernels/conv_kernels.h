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

class ConvOperator : public habana::HabanaOperator {
 public:
  ConvOperator(int device_id, c10::ScalarType scalarType);
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);

  virtual void SetPTOutputs(torch::jit::Stack& inputs);

 private:
  c10::ScalarType scalarType_;
  std::vector<synapse_helpers::tensor_or_ref> tensors_;
};

//
// For suporting conv2d operation from Graph mode
class Conv2dOperator : public ConvOperator {
 public:
  Conv2dOperator(int device_id, c10::ScalarType scalarType)
      : ConvOperator(device_id, scalarType) {}
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent) {
    TORCH_CHECK(
        inputs.size() == 7, "Conv2d Operation expects 7 arguments as input")
    bool transposed = false;
    c10::IntArrayRef output_padding = {0, 0, 0, 0};
    inputs.insert(inputs.begin() + 6, c10::IValue(transposed));
    inputs.insert(inputs.begin() + 7, c10::IValue(output_padding));
    ConvOperator::AllocateAndAddSynapseNode(
        graph, inputs, is_output_persistent);
  }
};

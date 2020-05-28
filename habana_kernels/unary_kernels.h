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
using namespace habana;

// Unary Operator
//
class UnaryOperator : public HabanaOperator {
 public:
  UnaryOperator(int device_id, const std::string& guid);
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);
};

//
// Relu Operator
class ReluOperator : public UnaryOperator {
 public:
  ReluOperator(int device_id, c10::ScalarType scalarType);
};

//
// Sigmoid Operator
class SigmoidOperator : public UnaryOperator {
 public:
  SigmoidOperator(int device_id, c10::ScalarType scalarType);
};
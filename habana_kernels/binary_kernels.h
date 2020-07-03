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
#include "habana_kernels/tensor_shape_kernels.h"

namespace habana {

class BinaryOperator : public habana::HabanaOperator {
 public:
  BinaryOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    // TODO: add meta data for broadcasting in graph mode
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);
  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

class MulOperator : public BinaryOperator {
 public:
  // Mul op
  MulOperator(int device_id, c10::ScalarType scalarType)
      : BinaryOperator(
            device_id,
            "mult_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

class DivOperator : public BinaryOperator {
 public:
  DivOperator(int device_id, c10::ScalarType scalarType)
      : BinaryOperator(
            device_id,
            "div_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

class AddOperator : public habana::HabanaOperator {
 public:
  AddOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "add_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    scalarType_ = scalarType;
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);
  void insert_reshape_op(
      synapse_helpers::graph& graph,
      ReshapeOperator reshapeOp,
      at::Tensor& arg,
      int position,
      int64_t out_dims);
  virtual void SetPTOutputs(torch::jit::Stack& inputs);

 private:
  c10::ScalarType scalarType_;
};

} // namespace habana

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
#include "habana_kernels/simple_generic_kernel.h"
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
  void insert_reshape_op(
      synapse_helpers::graph& graph,
      ReshapeOperator& reshapeOp,
      at::Tensor& arg,
      int32_t position,
      int64_t out_dims);

 protected:
  c10::ScalarType scalarType_;
};

class MulOperator : public BinaryOperator {
 public:
  // Mul op
  MulOperator(int device_id, c10::ScalarType scalarType)
      : BinaryOperator(
            device_id,
            "mult_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

class DivOperator : public BinaryOperator {
 public:
  DivOperator(int device_id, c10::ScalarType scalarType)
      : BinaryOperator(
            device_id,
            "div_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

class PowOperator : public BinaryOperator {
 public:
  PowOperator(int device_id, c10::ScalarType scalarType)
      : BinaryOperator(
            device_id,
            "pow_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

class BinaryOperatorWithAlpha : public BinaryOperator {
 public:
  BinaryOperatorWithAlpha(int device_id, const std::string& guid)
      : BinaryOperator(device_id, guid) {}
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);
};

class AddOperator : public BinaryOperatorWithAlpha {
 public:
  AddOperator(int device_id, c10::ScalarType scalarType)
      : BinaryOperatorWithAlpha(
            device_id,
            "add_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

class SubOperator : public BinaryOperatorWithAlpha {
 public:
  SubOperator(int device_id, c10::ScalarType scalarType)
      : BinaryOperatorWithAlpha(
            device_id,
            "sub_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

} // namespace habana

// The following function definitions are added because compare_kernels.cpp
// has dependence on these.
at::Tensor convert_scalar_to_tensor_using_self(
    const at::Tensor& self,
    c10::Scalar other);

void do_generic_tensor_binary_op_out(
    at::Tensor& output,
    const at::Tensor& operand1,
    const at::Tensor& operand2,
    const std::string& op,
    SynapsePassType pass_type);

at::Tensor get_correct_input_tensor(const at::Tensor& arg1, const at::Tensor& arg2);

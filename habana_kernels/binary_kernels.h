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
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"

namespace habana {

class BinaryOperator : public habana::HabanaOperator {
 public:
  BinaryOperator(
      int device_id,
      const std::string& guid,
      c10::ScalarType scalarType)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    scalarType_ = scalarType;
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
  void insert_reshape_op(
      synapse_helpers::graph& graph,
      ReshapeOperator& reshapeOp,
      at::Tensor& arg,
      int32_t position,
      int64_t out_dims);

 protected:
  c10::ScalarType scalarType_;
};

class BinaryWrapperOperator : public habana::HabanaOperator {
 public:
  BinaryWrapperOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) final;
  void SetPTOutputs(torch::jit::Stack& inputs);

 protected:
  c10::ScalarType scalarType_;
};

class MulOperator : public BinaryWrapperOperator {
 public:
  // Mul op
  MulOperator(int device_id, c10::ScalarType scalarType)
      : BinaryWrapperOperator(
            device_id,
            "mult_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

class DivOperator : public BinaryWrapperOperator {
 public:
  DivOperator(int device_id, c10::ScalarType scalarType)
      : BinaryWrapperOperator(
            device_id,
            "div_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

class PowOperator : public BinaryWrapperOperator {
 public:
  PowOperator(int device_id, c10::ScalarType scalarType)
      : BinaryWrapperOperator(
            device_id,
            "pow_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

class BinaryOperatorWithAlpha : public BinaryOperator {
 public:
  BinaryOperatorWithAlpha(
      int device_id,
      const std::string& guid,
      c10::ScalarType scalarType)
      : BinaryOperator(device_id, guid, scalarType) {}
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) final;
};

class BinaryWrapperOperatorWithAlpha : public habana::HabanaOperator {
 public:
  BinaryWrapperOperatorWithAlpha(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) final;
  void SetPTOutputs(torch::jit::Stack& inputs);

 protected:
  c10::ScalarType scalarType_;
};

class AddOperator : public BinaryWrapperOperatorWithAlpha {
 public:
  AddOperator(int device_id, c10::ScalarType scalarType)
      : BinaryWrapperOperatorWithAlpha(
            device_id,
            "add_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

class SubOperator : public BinaryWrapperOperatorWithAlpha {
 public:
  SubOperator(int device_id, c10::ScalarType scalarType)
      : BinaryWrapperOperatorWithAlpha(
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

at::Tensor get_correct_input_tensor(
    const at::Tensor& arg1,
    const at::Tensor& arg2);

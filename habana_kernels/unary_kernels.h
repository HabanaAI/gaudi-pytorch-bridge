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
class UnaryOperator : public HabanaOperator {
 public:
  UnaryOperator(int device_id, const std::string& guid) : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);
};

// Unary Backward Operator
class UnaryBackwardOperator : public HabanaOperator {
 public:
  UnaryBackwardOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);
};

// Relu Operator
class ReluOperator : public UnaryOperator {
 public:
  ReluOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "relu_fwd_" + habana_helpers::name_suffix_from_type(scalarType)){};
};

// Relu Operator
class ReluInplaceOperator : public HabanaOperator {
 public:
  ReluInplaceOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

// Sigmoid Operator
class SigmoidOperator : public UnaryOperator {
 public:
  SigmoidOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "sigmoid_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)){};
};

// SigmoidBackward Operator
class SigmoidBackwardOperator : public UnaryBackwardOperator {
 public:
  SigmoidBackwardOperator(int device_id, c10::ScalarType scalarType)
      : UnaryBackwardOperator(
            device_id,
            "sigmoid_bwd_" +
                habana_helpers::name_suffix_from_type(scalarType)){};
};

// Tanh Operator
class TanhOperator : public UnaryOperator {
 public:
  TanhOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "tanh_fwd_" + habana_helpers::name_suffix_from_type(scalarType)){};
};

// TanhBackward Operator
class TanhBackwardOperator : public UnaryBackwardOperator {
 public:
  TanhBackwardOperator(int device_id, c10::ScalarType scalarType)
      : UnaryBackwardOperator(
            device_id,
            "tanh_bwd_" + habana_helpers::name_suffix_from_type(scalarType)){};
};

// Abs Operator
class AbsOperator : public UnaryOperator {
 public:
  AbsOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "abs_fwd_" + habana_helpers::name_suffix_from_type(scalarType)){};
};

// Sqrt Operator
class SqrtOperator : public UnaryOperator {
 public:
  SqrtOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "sqrt_fwd_" + habana_helpers::name_suffix_from_type(scalarType)){};
};

// Clamp Operator
class ClampOperator : public HabanaOperator {
 public:
  ClampOperator(int device_id, const std::string& guid) : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

// Neg Operator
class NegOperator : public UnaryOperator {
 public:
  NegOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "neg_fwd_" + habana_helpers::name_suffix_from_type(scalarType)){};
};
//
// ReciprocalOut Operator
class ReciprocalOutOperator : public HabanaOperator {
 public:
  ReciprocalOutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "reciprocal_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

//
// Reciprocal Operator
class ReciprocalOperator : public ReciprocalOutOperator {
 public:
  ReciprocalOperator(int device_id, c10::ScalarType scalarType)
      : ReciprocalOutOperator(device_id, scalarType) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

//
// Gelu Operator
class GeluOperator : public HabanaOperator {
 public:
  GeluOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "gelu_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

//
// Erf Operator
class ErfOperator : public HabanaOperator {
 public:
  ErfOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "erf_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

//
// Exp Operator
class ExpOperator : public HabanaOperator {
 public:
  ExpOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "exp_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};


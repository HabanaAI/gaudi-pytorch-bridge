/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#pragma once
#include "backend/habana_operator.h"
namespace habana {

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
      const OutputMetaDataVector& output_metadata) override;

  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;
};

// Tanh Operator
class TanhOperator : public UnaryOperator {
 public:
  TanhOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "tanh_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Abs Operator
class AbsOperator : public UnaryOperator {
 public:
  AbsOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "abs_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Sqrt Operator
class SqrtOperator : public UnaryOperator {
 public:
  SqrtOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "sqrt_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
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
      const OutputMetaDataVector& output_metadata) override;
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
      const OutputMetaDataVector& output_metadata) override;
};

//
// HbGelu Operator
class HbGeluOperator : public HabanaOperator {
 public:
  HbGeluOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "gelu_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void SetEagerMode() {
    is_eager_mode = true;
  }

  bool isEagerMode() const {
    return is_eager_mode;
  }

 private:
  bool is_eager_mode = false;
};

//
// Gelu Backward Operator
class GeluBackwardOperator : public HabanaOperator {
 public:
  GeluBackwardOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "gelu_bwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Exp Operator
class ExpOperator : public UnaryOperator {
 public:
  ExpOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "exp_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

class LogOperator : public UnaryOperator {
 public:
  LogOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "log_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};
} // namespace habana

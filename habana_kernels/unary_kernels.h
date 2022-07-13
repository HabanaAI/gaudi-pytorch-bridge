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
      const OutputMetaDataVector& output_metadata);
};

class UnaryInplaceOperator : public HabanaOperator {
 public:
  UnaryInplaceOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata);
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
      const OutputMetaDataVector& output_metadata);
};

// Wrapper of unary which produces one tensor output and accepts
// one tensor and scalar(s) as inputs.
class UnaryLikeOperator : public UnaryOperator {
 public:
  UnaryLikeOperator(
      int device_id,
      const std::string& guid,
      bool inplace = false)
      : UnaryOperator(device_id, guid), m_inplace{inplace} {}

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata);

 protected:
  bool m_inplace;
};

// Relu Operator
class ReluOperator : public UnaryOperator {
 public:
  ReluOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "relu_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Relu Inplace Operator
class ReluInplaceOperator : public UnaryInplaceOperator {
 public:
  ReluInplaceOperator(int device_id, c10::ScalarType scalarType)
      : UnaryInplaceOperator(
            device_id,
            "relu_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Leaky Relu Operator
class LeakyReluOperator : public UnaryLikeOperator {
 public:
  LeakyReluOperator(
      int device_id,
      c10::ScalarType scalarType,
      bool inplace = false)
      : UnaryLikeOperator(
            device_id,
            "leakyrelu_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType),
            inplace) {}
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Elu Operator
class EluOperator : public UnaryLikeOperator {
 public:
  EluOperator(int device_id, c10::ScalarType scalarType, bool inplace = false)
      : UnaryLikeOperator(
            device_id,
            "elu_fwd_" + habana_helpers::name_suffix_from_type(scalarType),
            inplace) {}

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class LeakyReluBackwardOperator : public UnaryBackwardOperator {
 public:
  LeakyReluBackwardOperator(int device_id, c10::ScalarType scalarType)
      : UnaryBackwardOperator(
            device_id,
            "leakyrelu_bwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Sigmoid Operator
class SigmoidOperator : public UnaryOperator {
 public:
  SigmoidOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "sigmoid_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
};

// SigmoidBackward Operator
class SigmoidBackwardOperator : public UnaryBackwardOperator {
 public:
  SigmoidBackwardOperator(int device_id, c10::ScalarType scalarType)
      : UnaryBackwardOperator(
            device_id,
            "sigmoid_bwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
};

class HardsigmoidOperator : public UnaryLikeOperator {
 public:
  HardsigmoidOperator(
      int device_id,
      c10::ScalarType scalarType,
      bool inplace = false)
      : UnaryLikeOperator(
            device_id,
            "hard_sigmoid_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType),
            inplace) {}
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class HardsigmoidBackwardOperator : public UnaryBackwardOperator {
 public:
  HardsigmoidBackwardOperator(int device_id, c10::ScalarType scalarType)
      : UnaryBackwardOperator(
            device_id,
            "hard_sigmoid_bwd_" +
                habana_helpers::name_suffix_from_type(scalarType)){};
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Tanh Operator
class TanhOperator : public UnaryOperator {
 public:
  TanhOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "tanh_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// TanhBackward Operator
class TanhBackwardOperator : public UnaryBackwardOperator {
 public:
  TanhBackwardOperator(int device_id, c10::ScalarType scalarType)
      : UnaryBackwardOperator(
            device_id,
            "tanh_bwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

class SiluBackwardOperator : public UnaryBackwardOperator {
 public:
  SiluBackwardOperator(int device_id, c10::ScalarType scalarType)
      : UnaryBackwardOperator(device_id, NULL_GUID) {
    static_cast<void>(scalarType);
  }

  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Abs Operator
class AbsOperator : public UnaryOperator {
 public:
  AbsOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "abs_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Abs Inplace Operator

class AbsInplaceOperator : public UnaryInplaceOperator {
 public:
  AbsInplaceOperator(int device_id, c10::ScalarType scalarType)
      : UnaryInplaceOperator(
            device_id,
            "abs_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Round Operator
class RoundOperator : public UnaryOperator {
 public:
  RoundOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "round_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Round Inplace Operator
class RoundInplaceOperator : public UnaryInplaceOperator {
 public:
  RoundInplaceOperator(int device_id, c10::ScalarType scalarType)
      : UnaryInplaceOperator(
            device_id,
            "round_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Rsqrt Operator
class RsqrtOperator : public UnaryOperator {
 public:
  RsqrtOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "rsqrt_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Rsqrt Inplace Operator
class RsqrtInplaceOperator : public UnaryInplaceOperator {
 public:
  RsqrtInplaceOperator(int device_id, c10::ScalarType scalarType)
      : UnaryInplaceOperator(
            device_id,
            "rsqrt_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Isfinite Operator
class IsfiniteOperator : public UnaryOperator {
 public:
  IsfiniteOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "isfinite_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Sqrt Operator
class SqrtOperator : public UnaryOperator {
 public:
  SqrtOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "sqrt_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Clamp Operator
class ClampOperator : public UnaryOperator {
 public:
  ClampOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "clamp_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class ClampInplaceOperator : public UnaryOperator {
 public:
  ClampInplaceOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "clamp_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class ClampMinOperator : public UnaryOperator {
 public:
  ClampMinOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "clamp_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Neg Operator
class NegOperator : public UnaryOperator {
 public:
  NegOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "neg_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Sin Operator
class SinOperator : public UnaryOperator {
 public:
  SinOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "sin_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Cos Operator
class CosOperator : public UnaryOperator {
 public:
  CosOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "cos_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
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
// reciprocal Inplace Operator
class ReciprocalInplaceOperator : public UnaryInplaceOperator {
 public:
  ReciprocalInplaceOperator(int device_id, c10::ScalarType scalarType)
      : UnaryInplaceOperator(
            device_id,
            "reciprocal_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
};

//
// Gelu Operator
class GeluOperator : public HabanaOperator {
 public:
  GeluOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "gelu_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
    p_context_->excluded_output_indices_ = {1};
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

//
// Erf Operator
class ErfOperator : public UnaryOperator {
 public:
  ErfOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "erf_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Exp Operator
class ExpOperator : public UnaryOperator {
 public:
  ExpOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "exp_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Erf Inplace Operator
class ErfInplaceOperator : public UnaryInplaceOperator {
 public:
  ErfInplaceOperator(int device_id, c10::ScalarType scalarType)
      : UnaryInplaceOperator(
            device_id,
            "erf_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Exp Inplace Operator
class ExpInplaceOperator : public UnaryInplaceOperator {
 public:
  ExpInplaceOperator(int device_id, c10::ScalarType scalarType)
      : UnaryInplaceOperator(
            device_id,
            "exp_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

class SqrtInplaceOperator : public UnaryInplaceOperator {
 public:
  SqrtInplaceOperator(int device_id, c10::ScalarType scalarType)
      : UnaryInplaceOperator(
            device_id,
            "sqrt_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Floor Operator
class FloorOperator : public UnaryOperator {
 public:
  FloorOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "floor_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Floor Inplace Operator
class FloorInplaceOperator : public UnaryInplaceOperator {
 public:
  FloorInplaceOperator(int device_id, c10::ScalarType scalarType)
      : UnaryInplaceOperator(
            device_id,
            "floor_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

class LogOperator : public UnaryOperator {
 public:
  LogOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "log_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

class LogInplaceOperator : public UnaryInplaceOperator {
 public:
  LogInplaceOperator(int device_id, c10::ScalarType scalarType)
      : UnaryInplaceOperator(
            device_id,
            "log_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

class Log2Operator : public UnaryOperator {
 public:
  Log2Operator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "log2_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

class Log2InplaceOperator : public UnaryInplaceOperator {
 public:
  Log2InplaceOperator(int device_id, c10::ScalarType scalarType)
      : UnaryInplaceOperator(
            device_id,
            "log2_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

// isnan Operator
class IsnanOperator : public UnaryOperator {
 public:
  IsnanOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "isnan_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// SiluOut Operator
class SiluOutOperator : public HabanaOperator {
 public:
  SiluOutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "sigmoid_fwd_" +
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

// Silu Operator
class SiluOperator : public UnaryOperator {
 public:
  SiluOperator(int device_id, c10::ScalarType scalarType)
      : UnaryOperator(
            device_id,
            "sigmoid_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Cumsum Operator
class CumsumOperator : public UnaryLikeOperator {
 public:
  CumsumOperator(int device_id, c10::ScalarType scalarType)
      : UnaryLikeOperator(
            device_id,
            "cumsum_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

} // namespace habana

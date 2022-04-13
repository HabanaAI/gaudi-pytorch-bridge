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

at::Tensor get_hpu_tensor(at::Tensor input);

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
      const OutputMetaDataVector& output_metadata) override;

  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& arg1,
      const at::Tensor& arg2);

 protected:
  c10::ScalarType scalarType_;

  bool MaybeMultiplyWithBool(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaData& output_metadata);
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
      const OutputMetaDataVector& output_metadata) final;
  void SetPTOutputs(torch::jit::Stack& inputs);

  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;

 protected:
  c10::ScalarType scalarType_;
};

class MulOperator : public BinaryWrapperOperator {
 public:
  // Mul op
  MulOperator(int device_id, c10::ScalarType scalarType)
      : BinaryWrapperOperator(
            device_id,
            MULT_GUID + habana_helpers::name_suffix_from_type(scalarType)) {
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
      const OutputMetaDataVector& output_metadata) final;
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
      const OutputMetaDataVector& output_metadata) override;
  void SetPTOutputs(torch::jit::Stack& inputs) override;

  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;

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

class RsubOperator : public SubOperator {
 public:
  RsubOperator(int device_id, c10::ScalarType scalarType)
      : SubOperator(device_id, scalarType) {}
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;
};

class MaximumOperator : public BinaryWrapperOperator {
 public:
  // Maximum op
  MaximumOperator(int device_id, c10::ScalarType scalarType)
      : BinaryWrapperOperator(
            device_id,
            "max_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

class MinimumOperator : public BinaryWrapperOperator {
 public:
  // Minimum op
  MinimumOperator(int device_id, c10::ScalarType scalarType)
      : BinaryWrapperOperator(
            device_id,
            "min_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

class RemainderOperator : public HabanaOperator {
 public:
  RemainderOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "div_mod_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class RemainderWrapperOperator : public HabanaOperator {
 public:
  RemainderWrapperOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "div_mod_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    this->scalarType_ = scalarType;
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;

 protected:
  c10::ScalarType scalarType_;
};

class RemainderInplaceOperator : public habana::HabanaOperator {
 public:
  RemainderInplaceOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "div_mod_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    scalarType_ = scalarType;
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

 protected:
  c10::ScalarType scalarType_;
};

class RemainderInplaceWrapperOperator : public habana::HabanaOperator {
 public:
  RemainderInplaceWrapperOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "div_mod_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    scalarType_ = scalarType;
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;

 protected:
  c10::ScalarType scalarType_;
};

class RemainderOutOperator : public HabanaOperator {
 public:
  RemainderOutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "div_mod_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class RemainderOutWrapperOperator : public HabanaOperator {
 public:
  RemainderOutWrapperOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "div_mod_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    this->scalarType_ = scalarType;
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

 protected:
  c10::ScalarType scalarType_;
};
} // namespace habana

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
#include "habana_kernels/tensor_shape_kernels.h"

namespace habana {

class BinaryInplaceOperator : public habana::HabanaOperator {
 public:
  BinaryInplaceOperator(
      int device_id,
      const std::string& guid,
      c10::ScalarType scalarType)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    scalarType_ = scalarType;
  }
  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

 protected:
  c10::ScalarType scalarType_;
};

class BinaryInplaceWrapperOperator : public habana::HabanaOperator {
 public:
  BinaryInplaceWrapperOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }
  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;

 protected:
  c10::ScalarType scalarType_;
};

class MulInplaceOperator : public BinaryInplaceWrapperOperator {
 public:
  // Mul op
  MulInplaceOperator(int device_id, c10::ScalarType scalarType)
      : BinaryInplaceWrapperOperator(
            device_id,
            MULT_GUID + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::ANY});
  }
};

class DivInplaceOperator : public BinaryInplaceWrapperOperator {
 public:
  // Div op
  DivInplaceOperator(int device_id, c10::ScalarType scalarType)
      : BinaryInplaceWrapperOperator(
            device_id,
            "div_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

class PowInplaceOperator : public BinaryInplaceWrapperOperator {
 public:
  // Pow op
  PowInplaceOperator(int device_id, c10::ScalarType scalarType)
      : BinaryInplaceWrapperOperator(
            device_id,
            "pow_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

class BinaryInplaceOperatorWithAlpha : public BinaryInplaceOperator {
 public:
  BinaryInplaceOperatorWithAlpha(
      int device_id,
      const std::string& guid,
      c10::ScalarType scalarType)
      : BinaryInplaceOperator(device_id, guid, scalarType) {}
  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;
};

class BinaryInplaceWrapperOperatorWithAlpha : public habana::HabanaOperator {
 public:
  BinaryInplaceWrapperOperatorWithAlpha(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }
  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

 protected:
  c10::ScalarType scalarType_;
};

class AddInplaceOperator : public BinaryInplaceWrapperOperatorWithAlpha {
 public:
  AddInplaceOperator(int device_id, c10::ScalarType scalarType)
      : BinaryInplaceWrapperOperatorWithAlpha(
            device_id,
            "add_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

class SubInplaceOperator : public BinaryInplaceWrapperOperatorWithAlpha {
 public:
  SubInplaceOperator(int device_id, c10::ScalarType scalarType)
      : BinaryInplaceWrapperOperatorWithAlpha(
            device_id,
            "sub_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    scalarType_ = scalarType;
  }
};

class AddcmulInplaceOperator : public habana::HabanaOperator {
 public:
  AddcmulInplaceOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("addcmul_fwd_") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class AddcdivInplaceOperator : public habana::HabanaOperator {
 public:
  AddcdivInplaceOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("addcdiv_fwd_") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};
} // namespace habana

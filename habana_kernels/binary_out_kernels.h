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

class BinaryOutOperator : public habana::HabanaOperator {
 public:
  BinaryOutOperator(
      int device_id,
      const std::string& guid,
      c10::ScalarType scalarType)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    scalarType_ = scalarType;
  }
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& arg1,
      const at::Tensor& arg2);

 protected:
  c10::ScalarType scalarType_;
};

class MulOutOperator : public BinaryOutOperator {
 public:
  // Mul op
  MulOutOperator(int device_id, c10::ScalarType scalarType)
      : BinaryOutOperator(
            device_id,
            MULT_GUID + habana_helpers::name_suffix_from_type(scalarType),
            scalarType) {}
};

class DivOutOperator : public BinaryOutOperator {
 public:
  // Div op
  DivOutOperator(int device_id, c10::ScalarType scalarType)
      : BinaryOutOperator(
            device_id,
            "div_fwd_" + habana_helpers::name_suffix_from_type(scalarType),
            scalarType) {}
};

// TODO add wrapper for add and sub out varient

} // namespace habana

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

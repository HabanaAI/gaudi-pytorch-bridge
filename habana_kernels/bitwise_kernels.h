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

class BitwiseOutOperator : public HabanaOperator {
 public:
  BitwiseOutOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& arg1,
      const at::Tensor& arg2);
};

class BitwiseOutWrapOperator : public HabanaOperator {
 public:
  BitwiseOutWrapOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

class BitwiseAndOutOperator : public BitwiseOutWrapOperator {
 public:
  BitwiseAndOutOperator(int device_id, c10::ScalarType scalarType)
      : BitwiseOutWrapOperator(
            device_id,
            "and_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

class BitwiseOrOutOperator : public BitwiseOutWrapOperator {
 public:
  BitwiseOrOutOperator(int device_id, c10::ScalarType scalarType)
      : BitwiseOutWrapOperator(
            device_id,
            "or_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};
class BitwiseXorOutOperator : public BitwiseOutWrapOperator {
 public:
  BitwiseXorOutOperator(int device_id, c10::ScalarType scalarType)
      : BitwiseOutWrapOperator(
            device_id,
            "xor_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};
class BitwiseNotOutOperator : public HabanaOperator {
 public:
  BitwiseNotOutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "not_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
  }
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};
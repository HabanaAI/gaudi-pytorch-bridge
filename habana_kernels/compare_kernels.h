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
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/habana_operator.h"

class CompareOutOperator : public habana::HabanaOperator {
 public:
  CompareOutOperator(
      int device_id,
      c10::ScalarType scalarType,
      const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    this->scalarType_ = scalarType;
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

 protected:
  c10::ScalarType scalarType_;
};

class CompareOperator : public CompareOutOperator {
 public:
  CompareOperator(
      int device_id,
      c10::ScalarType scalarType,
      const std::string& guid)
      : CompareOutOperator(device_id, scalarType, guid) {}
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

class GtOperator : public CompareOperator {
 public:
  GtOperator(int device_id, c10::ScalarType scalarType)
      : CompareOperator(
            device_id,
            scalarType,
            "greater_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
};

class EqOutOperator : public CompareOutOperator {
 public:
  EqOutOperator(int device_id, c10::ScalarType scalarType)
      : CompareOutOperator(
            device_id,
            scalarType,
            "equal_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
};

class EqOperator : public CompareOperator {
 public:
  EqOperator(int device_id, c10::ScalarType scalarType)
      : CompareOperator(
            device_id,
            scalarType,
            "equal_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
};
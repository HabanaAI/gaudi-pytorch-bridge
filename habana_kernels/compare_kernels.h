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

namespace habana {
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

class CompareOutWrapperOperator : public habana::HabanaOperator {
 public:
  CompareOutWrapperOperator(
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

  void SetPTOutputs(torch::jit::Stack& inputs) override;

 protected:
  c10::ScalarType scalarType_;
};

class CompareWrapperOperator : public CompareOutWrapperOperator {
 public:
  CompareWrapperOperator(
      int device_id,
      c10::ScalarType scalarType,
      const std::string& guid)
      : CompareOutWrapperOperator(device_id, scalarType, guid) {}

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) final;

  void SetPTOutputs(torch::jit::Stack& inputs);

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& arg1,
      const at::Tensor& arg2);
};

class GtOperator : public CompareWrapperOperator {
 public:
  GtOperator(int device_id, c10::ScalarType scalarType)
      : CompareWrapperOperator(
            device_id,
            scalarType,
            "greater_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
};

class EqOutOperator : public CompareOutWrapperOperator {
 public:
  EqOutOperator(int device_id, c10::ScalarType scalarType)
      : CompareOutWrapperOperator(
            device_id,
            scalarType,
            "equal_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

class EqOperator : public CompareWrapperOperator {
 public:
  EqOperator(int device_id, c10::ScalarType scalarType)
      : CompareWrapperOperator(
            device_id,
            scalarType,
            "equal_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

class LtOperator : public CompareWrapperOperator {
 public:
  LtOperator(int device_id, c10::ScalarType scalarType)
      : CompareWrapperOperator(
            device_id,
            scalarType,
            "less_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {}
};

class GeOperator : public CompareWrapperOperator {
 public:
  GeOperator(int device_id, c10::ScalarType scalarType)
      : CompareWrapperOperator(
            device_id,
            scalarType,
            "greater_equal_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Ge Operator
class GeOutOperator : public habana::HabanaOperator {
 public:
  GeOutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "greater_equal_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    scalarType_ = scalarType;
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

 protected:
  c10::ScalarType scalarType_;
};

class LeOperator : public CompareWrapperOperator {
 public:
  LeOperator(int device_id, c10::ScalarType scalarType)
      : CompareWrapperOperator(
            device_id,
            scalarType,
            "less_equal_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
};

// Dummy class added to capture only meta-data information about this operator
// for JIT passes. Do not instantiate objects of this class.
class NeOperator : public habana::HabanaOperator {
 public:
  NeOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "ne_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
  }
};

} // namespace habana

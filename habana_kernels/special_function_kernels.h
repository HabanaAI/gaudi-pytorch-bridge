/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
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
// Special Operator
class SpecialFunctionOperator : public HabanaOperator {
 public:
  SpecialFunctionOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) = 0;
  virtual ~SpecialFunctionOperator(){};
};

// TODO: Use this function globally
std::string getGUID(
    const char* op_code,
    bool inplace,
    c10::ScalarType scalarType) {
  std::string guid(op_code);
  // For non inplace, op_code doesn't have "_" suffix
  // Hence added
  if (!inplace) {
    guid += "_";
  }

  guid += "fwd_";
  guid += habana_helpers::name_suffix_from_type(scalarType);
  return guid;
}

class SpecialFunctionFwdOperator : public SpecialFunctionOperator {
 public:
  SpecialFunctionFwdOperator(
      int device_id,
      c10::ScalarType scalarType,
      const char* op_code,
      bool inplace = false)
      : SpecialFunctionOperator(
            device_id,
            getGUID(op_code, inplace, scalarType)),
        m_inplace{inplace} {};
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

 private:
  bool m_inplace;
};
} // namespace habana

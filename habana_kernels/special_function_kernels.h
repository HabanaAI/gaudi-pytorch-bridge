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
using namespace habana;
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
      bool is_output_persistent = false);
};

// Asin Operator
std::string getString(const char* op_code) {
  return std::string(op_code);
}
class SpecialFunctionFwdOperator : public SpecialFunctionOperator {
 public:
  SpecialFunctionFwdOperator(
      int device_id,
      c10::ScalarType scalarType,
      const char* op_code)
      : SpecialFunctionOperator(
            device_id,
            getString(op_code) + "_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)){};
};

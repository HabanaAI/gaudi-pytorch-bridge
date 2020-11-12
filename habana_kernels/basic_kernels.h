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
#include <perf_lib_layer_params.h>
#include "habana_kernels/habana_operator.h"

//
// ToDtype Operator
class ToDtypeOperator : public habana::HabanaOperator {
 public:
  ToDtypeOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("to_dtype") {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  // virtual void SetPTOutput(torch::jit::Stack& inputs) override;
};

//
// Cast Operator (Lazy mode only)
class CastLazyOperator : public habana::HabanaOperator {
 public:
  CastLazyOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("cast_lazy") {
    this->CreateSynContext(device_id);
    kernel_meta_data_.tpc_input_order = {0};
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  /**
   * @brief CastKernel params structure
   */
  ns_CastKernel::Params synapse_cast_params_builder() {
    ns_CastKernel::Params cast_params{};
    cast_params.round_mode = CAST_ROUND_HALF_NE;
    return cast_params;
  }
};

//
// MemCopy Operator
class MemCopyOperator : public habana::HabanaOperator {
 public:
  MemCopyOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("memcpy") {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

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
using namespace torch;

namespace habana {

class UpsampleOperator : public HabanaOperator {
 public:
  UpsampleOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::NHWC});
    kernel_meta_data_.output_layout.assign({LayoutFormat::NHWC});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);
  virtual void SetPTOutputs(torch::jit::Stack& inputs);
  static std::vector<int64_t> compute_output_shape(
      std::vector<int64_t> shape_in,
      c10::optional<IntArrayRef> output_size,
      c10::optional<at::ArrayRef<double>> scales,
      c10::MemoryFormat memory_format);
};

// Upsample Backward Operator
class UpsampleBackwardOperator : public HabanaOperator {
 public:
  UpsampleBackwardOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::NHWC});
    kernel_meta_data_.output_layout.assign({LayoutFormat::NHWC});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);
  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

class UpsampleNearest2dOperator : public UpsampleOperator {
 public:
  UpsampleNearest2dOperator(int device_id, c10::ScalarType scalarType)
      : UpsampleOperator(
            device_id,
            "upsample_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
};

class UpsampleNearest2dBackwardOperator : public UpsampleBackwardOperator {
 public:
  UpsampleNearest2dBackwardOperator(int device_id, c10::ScalarType scalarType)
      : UpsampleBackwardOperator(
            device_id,
            "upsample_bwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
};

} // namespace habana

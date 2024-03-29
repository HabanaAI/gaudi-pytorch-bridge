/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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

#include <c10/util/ArrayRef.h>
#include <synapse_api.h>
#include <synapse_api_types.h>
#include <torch/script.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include "backend/synapse_helpers/habana_tensor.h"

#include <perf_lib_layer_params.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include "backend/habana_operator.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/graph.h"
#include "backend/synapse_helpers/recipe.h"
#include "habana_helpers/logging_pt.h"

namespace habana_helpers {
at::ScalarType getInternalDtype(at::ScalarType dtype);

std::optional<std::string> direct_cast_guid(
    std::pair<c10::ScalarType, c10::ScalarType> type_key);

// Check whether long is supported on Synapse side for given guid name
bool isLongTypeSupported(const std::string_view guid);

std::string_view GetPrecisionString(const c10::ScalarType& dtype);

void type_promotion_for_two_tensor_inputs(
    std::vector<at::IValue>& inputs,
    int& position_of_promoted_tensor,
    c10::ScalarType& compute_dtype,
    c10::ScalarType& dst_dtype);

void type_promotion_for_two_tensor_inputs(
    std::vector<at::IValue>& inputs,
    int& position_of_promoted_tensor,
    c10::ScalarType& compute_dtype);

std::vector<int64_t> compute_broadcast_shape(
    const at::Tensor& arg1,
    const at::Tensor& arg2);

void set_tensor_exp_bias(
    const at::Tensor& tensor,
    c10::optional<unsigned> exp_bias);

c10::optional<unsigned> get_tensor_exp_bias(const at::Tensor& tensor);

void set_output_hw_scaling_meta(
    const at::Tensor& input,
    const at::Tensor& output);

void set_output_hw_scaling_meta(
    const at::Tensor& input,
    habana::OutputMetaData& meta);

} // namespace habana_helpers

// CastOut Operator
class CastOutOperator : public habana::HabanaOperator {
 public:
  CastOutOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::ANY, habana::LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::ANY});
  }
  habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;

 protected:
  ns_CastKernel::Params synapse_cast_params_builder(
      c10::ScalarType dst_dtype,
      bool stochastic_rounding_override);
  ns_CastKernel::ParamsV2 synapse_cast_params_v2_builder(
      c10::ScalarType dst_dtype,
      bool stochastic_rounding_override,
      int seed);
};

// Cast Operator
class CastOperator : public CastOutOperator {
 public:
  CastOperator(int device_id, const std::string& guid)
      : CastOutOperator(device_id, guid) {
    kernel_meta_data_.input_layout.assign({habana::LayoutFormat::ANY});
  }
  habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
};

// Constant Operator
class ConstantOperator : public habana::HabanaOperator {
 public:
  ConstantOperator(int device_id, c10::ScalarType scalarType)
      : habana::HabanaOperator(
            habana::get_guid_with_precision("constant", scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::ANY});
  }

  habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
};

//
// For suporting ones_like operation
class OnesLikeOperator : public ConstantOperator {
 public:
  OnesLikeOperator(int device_id, c10::ScalarType scalarType)
      : ConstantOperator(device_id, scalarType) {}
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override {
    TORCH_CHECK(
        inputs.size() == 6,
        "OnesLikeOperator Operation expects 6 arguments as input")
    inputs.erase(inputs.begin() + 1, inputs.end());
    inputs.emplace_back(1);
    ConstantOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  }
};

// ConstantOut Operator
class ConstantOutOperator : public habana::HabanaOperator {
 public:
  ConstantOutOperator(int device_id, c10::ScalarType scalarType)
      : habana::HabanaOperator(
            habana::get_guid_with_precision("constant", scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::ANY});
    // special case, adding -1 to the tpc order, will not add any inputs
    kernel_meta_data_.tpc_input_order = {habana::NO_INPUTS};
  }

  habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
};

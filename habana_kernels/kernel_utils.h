/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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

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

#include "ATen/core/ivalue.h"
#include "generated/backend/amax.h"
#include "generated/backend/amin.h"
#include "generated/backend/aminmax.h"
#include "hpu_ops/backend/reduction_template.h"
#include "hpu_ops/shared_meta_common.h"

namespace sh = synapse_helpers;

namespace habana {

static std::tuple<at::Tensor, at::DimVector, bool> ParseSignature(
    const at::Stack& stack) {
  const at::Tensor& self = stack.at(0).toTensor();
  const bool is_dim_none = stack.size() == 1 || stack.at(1).isNone();
  const bool keepdim = stack.size() >= 3 && stack.at(2).toBool();

  at::DimVector dim_vec = is_dim_none
      ? at::DimVector{}
      : stack.at(1).isInt() ? at::DimVector{stack.at(1).toInt()}
                            : stack.at(1).toDimVector();
  return {self, dim_vec, keepdim};
}

static OutputMetaDataVector AminmaxMetaCommon(
    const at::Stack& stack,
    int count) {
  const auto [self, dim_vec, keepdim] = ParseSignature(stack);

  const auto shapes = ReductionOutputShape(self, dim_vec, keepdim);

  OutputMetaData meta;
  meta.shape = shapes[0];
  meta.dtype = self.scalar_type();
  return OutputMetaDataVector(count, meta);
}

OutputMetaDataVector AminmaxMeta(const at::Stack& stack) {
  return AminmaxMetaCommon(stack, 2);
}

OutputMetaDataVector AminAmaxMeta(const at::Stack& stack) {
  return AminmaxMetaCommon(stack, 1);
}

SharedMetaDataVector AmaxSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode) {
  return AminAmaxSharedMeta(stack, "reduce_max_multi_dim_fwd", executionMode);
}

SharedMetaDataVector AminSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode) {
  return AminAmaxSharedMeta(stack, "reduce_min_multi_dim_fwd", executionMode);
}

SharedMetaDataVector AminmaxSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode) {
  auto aminmaxSharedMeta =
      AminAmaxSharedMeta(stack, "reduce_min_multi_dim_fwd", executionMode);
  auto amaxSharedMeta =
      AminAmaxSharedMeta(stack, "reduce_max_multi_dim_fwd", executionMode);

  aminmaxSharedMeta.insert(
      aminmaxSharedMeta.end(), amaxSharedMeta.begin(), amaxSharedMeta.end());

  return aminmaxSharedMeta;
}

std::shared_ptr<void> FillAminAmaxParams(const at::Stack& stack, size_t& size) {
  const auto [self, dim_vec, keepdim] = ParseSignature(stack);
  auto rank = self.dim();

  PARAMS_STUB(ns_Reduction::ParamsV2);
  *params = FillReductionParams(rank, dim_vec, keepdim);

  return params;
}

void Aminmax::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  size_t paramsSize = 0;
  auto params = FillParams(stack, paramsSize);
  const auto meta = OutputMeta(stack)[0];

  c10::optional<sh::tensor> castedInput{};
  // Convert bool tensor to 0x00 and 0x01
  if (self.scalar_type() == c10::ScalarType::Bool) {
    castedInput = BuildBoolCast(
        this, graph, syn_in(0), self.sizes(), c10::ScalarType::Bool);
  }

  std::array<std::string, 2> guids = {
      get_guid_with_precision("reduce_min_multi_dim_fwd", meta.dtype),
      get_guid_with_precision("reduce_max_multi_dim_fwd", meta.dtype),
  };

  for (size_t i = 0; i < guids.size(); ++i) {
    // Leverage autocast feature from CGUID to support integer inputs
    if (c10::isIntegralType(meta.dtype, true)) {
      update_guid_dtype(guids[i], c10::ScalarType::Int);
    }

    auto input =
        castedInput.has_value() ? castedInput.value().get() : syn_in(0);
    auto op = BuildOp(
        graph,
        guids[i],
        {input},
        {{meta.shape, meta.dtype, i}},
        params.get(),
        paramsSize);
    syn_out(i) = std::move(op[0]);
  }
}

void AminAmax::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  size_t paramsSize = 0;
  auto params = FillParams(stack, paramsSize);
  const auto meta = OutputMeta(stack)[0];

  // Leverage autocast feature from CGUID to support integer inputs
  if (c10::isIntegralType(meta.dtype, true)) {
    update_guid_dtype(guid_, c10::ScalarType::Int);
  }

  c10::optional<sh::tensor> castedInput = c10::nullopt;
  // Convert bool tensor to 0x00 and 0x01
  if (self.scalar_type() == c10::ScalarType::Bool) {
    castedInput = BuildBoolCast(
        this, graph, syn_in(0), self.sizes(), c10::ScalarType::Bool);
  }

  auto input = castedInput.has_value() ? castedInput.value().get() : syn_in(0);
  auto op = BuildOp(
      graph,
      GetGuid(),
      {input},
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      paramsSize);
  syn_out(0) = std::move(op[0]);
}
} // namespace habana

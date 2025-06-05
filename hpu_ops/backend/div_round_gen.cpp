/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "hpu_ops/common/div_round_gen.h"
#include "generated/backend/div.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/binary_kernels.h"

namespace habana {

FillParamsT FillDivModeParams(const at::Stack& stack) {
  std::optional<std::string_view> rounding_mode =
      stack.at(2).toOptional<std::string_view>();
  if (rounding_mode.has_value()) {
    PARAMS_STUB(ns_DivModKernel::ParamsV2);
    const auto roundingModeSV = rounding_mode.value();
    params->isTruncRoundingMode = roundingModeSV == "trunc"sv;
    // Python div_mod is enabled where remainder returns the same sign of the
    // divisor, except for the zero remainder, which is enforced by the default
    // value 'true' for pyCompatible. Other value (false) is used for
    // div_rounding mode operator, for 'trunc' case.
    params->isPyCompatible = !(params->isTruncRoundingMode);
    return paramsT;
  } else
    return {};
}

OutputMetaDataVector DivModeMeta(const at::Stack& stack) {
  OutputMetaData meta{};
  const auto& self = stack_tensor(stack, 0);
  if (stack[1].isScalar()) {
    meta.shape = self.sizes().vec();
  } else {
    meta.shape = at::infer_size(self.sizes(), stack_tensor(stack, 1).sizes());
  }
  meta.dtype = GetResultDtype(stack, stack[2].isNone());

  return {meta};
}

SharedMetaDataVector DivModeSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto self = stack.at(0);
  auto selfTensor = self.toTensor();
  auto selfRank = selfTensor.dim();
  auto other = stack.at(1);
  int64_t otherRank = other.isTensor() ? other.toTensor().dim() : 1;
  auto outputRank = std::max(selfRank, otherRank);
  auto isRoundingModeNone = stack.at(2).isNone();
  auto commonType = GetCommonDtype(stack, isRoundingModeNone);
  auto resultType = GetResultDtype(stack, isRoundingModeNone);

  SharedMetaData divMeta{"round_divide_fwd"};
  divMeta.inputs_data = {{selfRank, commonType}, {otherRank, commonType}};
  divMeta.outputs_data = {{outputRank, resultType}};
  return {divMeta};
}

void RoundDivide::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  // When rounding_mode is none, then perform the operation in the default float
  // type
  if (stack.at(2).isNone()) {
    SetGuid(get_guid_with_precision(
        [] {
          using namespace std::literals;
          return "round_divide_fwd"sv;
        }(),
        DivModeMeta(stack)[0].dtype));
  }
  OpBackend::AddNode(graph, stack);
}
} // namespace habana

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

#include "generated/backend/mean.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {

namespace sh = synapse_helpers;

OutputMetaDataVector ReductionOpMeta(const at::Stack& stack) {
  return ReductionMeta<-1, -1, 1>(stack);
}

OutputMetaDataVector ReductionOpListMeta(const at::Stack& stack) {
  return ReductionMeta<1, 2, 3>(stack);
}

static bool areTypesAllowedForSumOut(
    const at::ScalarType& inputType,
    const at::ScalarType& outputType) {
  switch (inputType) {
    case at::kInt:
    case at::kFloat:
      return outputType == inputType;
    case at::kBFloat16:
    case at::kHalf:
    case at::kFloat8_e5m2:
    case at::kFloat8_e4m3fn:
      return outputType == inputType || outputType == at::kFloat;
    default:
      return false;
  }
}

static sh::tensor ReductionOpCommon(
    OpBackend* op,
    sh::graph& graph,
    synTensor input,
    const at::Stack& stack,
    const at::optional<uint8_t>& dim_index,
    const at::optional<uint8_t>& keepdim_index,
    const at::optional<uint8_t>& dtype_index) {
  auto self = stack_tensor(stack, 0);
  auto dtype = get_dtype(stack, dtype_index);

  // When the output tensor is provided, it indicates this is sum.out version,
  // and output tensor type should be used.
  const bool isSumOutVersion = stack.back().isTensor();
  if (isSumOutVersion) {
    dtype = stack.back().toTensor().scalar_type();
    if (dtype == at::kLong && !common::IsInt64Supported()) {
      dtype = at::kInt;
    } else if (dtype == at::kDouble) {
      dtype = at::kFloat;
    }
  }

  auto cast = HandleReductionDtype(op, graph, self, input, dtype);
  if (cast.has_value()) {
    input = cast.value().get();
  }

  auto dims = get_dims(stack, dim_index);
  bool keepdim = get_keepdim(stack, keepdim_index);

  int ndims = self.dim();
  auto params = FillReductionParams(ndims, dims, keepdim);
  auto shape = ReductionOutputShape(self, dims, keepdim)[0];

  // Due to dtype restrictions, a check is required to ensure the types are
  // allowed for the sum.out version.
  const bool isAdditionalCastNeeded =
      (isSumOutVersion &&
       !areTypesAllowedForSumOut(op->ScalarType(), dtype.value()));
  const c10::optional<int> finalResultIndex =
      isAdditionalCastNeeded ? c10::nullopt : c10::optional<int>(0);

  auto result = OpBackend::BuildNode(
      op,
      graph,
      {op->GetGuid(),
       {std::move(input)},
       {{shape, op->ScalarType(), finalResultIndex}},
       &params,
       sizeof(params)});

  if (isAdditionalCastNeeded) {
    auto resultCast = OpBackend::BuildCast(
        op, graph, result[0].get(), shape, op->ScalarType(), dtype.value(), 0);

    return resultCast;
  }

  return std::move(result[0]);
}

void ReductionOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  syn_out(0) = ReductionOpCommon(
      this, graph, syn_in(0), stack, c10::nullopt, c10::nullopt, 1);
}

void ReductionOpList::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  syn_out(0) = ReductionOpCommon(this, graph, syn_in(0), stack, 1, 2, 3);
}
} // namespace habana
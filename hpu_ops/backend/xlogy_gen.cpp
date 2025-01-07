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

#include "generated/backend/special_xlog1py.h"

namespace habana {

OutputMetaDataVector XlogYMeta(const at::Stack& stack) {
  OutputMetaData meta;
  c10::optional<at::Tensor> output_tensor = c10::nullopt;
  c10::optional<c10::ScalarType> output_type = c10::nullopt;
  auto size = stack.size();
  if (size > 2 && stack.at(size - 1).isTensor()) {
    output_tensor = stack.at(size - 1).toTensor();
    output_type = output_tensor.value().scalar_type();
  }

  auto self = stack_tensor(stack, 0);
  auto other = stack_tensor(stack, 1);
  meta.shape = at::infer_size(self.sizes(), other.sizes());
  meta.dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      output_tensor,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false,
      output_type);

  if (isIntegralType(meta.dtype, true)) {
    meta.dtype = torch::kFloat32;
  }

  return {meta};
}

bool ShouldCastToOutputType(
    c10::ScalarType dtype,
    c10::ScalarType output_dtype) {
  return isIntegralType(dtype, true) ||
      (dtype == at::kFloat && output_dtype == at::kBFloat16) ||
      (output_dtype == at::kFloat && dtype == at::kBFloat16);
}

SharedMetaDataVector XlogYSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto self = stack.at(0);
  auto other = stack.at(1);

  auto result_dtype = habana_helpers::DTypeHelper::get_compute_dtype(
    {self, other},
    c10::nullopt,
    habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteIntToFloat,
    false);

  unsigned selfDim = self.isTensor() ? self.toTensor().dim() : 1;
  unsigned otherDim = other.isTensor() ? other.toTensor().dim() : 1;
  unsigned outDim = std::max(selfDim, otherDim);

  SharedMetaData log1pMeta("log1p_fwd");
  log1pMeta.inputs_data.emplace_back(otherDim, result_dtype);
  log1pMeta.outputs_data.emplace_back(otherDim, result_dtype);

  SharedMetaData multMeta("mult");
  multMeta.inputs_data.emplace_back(selfDim, result_dtype);
  multMeta.inputs_data.emplace_back(otherDim, result_dtype);
  multMeta.outputs_data.emplace_back(outDim, result_dtype);

  return {log1pMeta, multMeta};
}

void Xlog1PyOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = XlogYMeta(stack)[0];
  auto other = stack_tensor(stack, 1);

  auto logy = BuildOp(
      graph,
      get_guid_with_precision("log1p_fwd", meta.dtype),
      {ShouldCastToOutputType(other.scalar_type(), meta.dtype)
           ? OpBackend::BuildCast(
                 this,
                 graph,
                 syn_in(1),
                 other.sizes().vec(),
                 other.scalar_type(),
                 meta.dtype)
                 .get()
           : syn_in(1)},
      {{other.sizes().vec(), meta.dtype}});

  auto xlogy = BuildOp(
      graph,
      get_guid_with_precision("mult", meta.dtype),
      {syn_in(0), logy[0].get()},
      {{meta.shape, meta.dtype, 0}});

  syn_out(0) = std::move(xlogy[0]);
}
} // namespace habana

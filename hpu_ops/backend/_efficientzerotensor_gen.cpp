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

#include "generated/backend/_efficientzerotensor.h"
#include "hpu_ops/op_backend.h"

namespace habana {

OutputMetaDataVector EfficientZeroMeta(const at::Stack& stack) {
  auto optionalDtype = stack.at(1).toOptional<at::ScalarType>();
  const at::ScalarType& type =
      optionalDtype.value_or(torch::get_default_dtype_as_scalartype());
  OutputMetaData meta{};

  meta.dtype = type;
  meta.shape = stack.at(0).toIntVector();

  return {meta};
}

SharedMetaDataVector EfficientZeroSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto rank = stack.at(0).toIntVector().size();
  auto optionalDtype = stack.at(1).toOptional<at::ScalarType>();
  const at::ScalarType& dtype =
      optionalDtype.value_or(torch::get_default_dtype_as_scalartype());

  SharedMetaData efficientZeroSharedMeta{"memset"};
  efficientZeroSharedMeta.outputs_data.emplace_back(rank, dtype);
  return {efficientZeroSharedMeta};
}

void EfficientZeroTensor::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = EfficientZeroMeta(stack)[0];
  syn_out(0) =
      std::move(BuildOp(graph, "memset", {}, {{meta.shape, meta.dtype, 0}})[0]);
}

} // namespace habana

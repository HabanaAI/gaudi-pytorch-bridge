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

#include "generated/backend/eye.h"

namespace habana {
OutputMetaDataVector EyeMeta(const at::Stack& stack) {
  OutputMetaData meta;
  const int64_t n = stack.at(0).toInt();
  if (stack.size() == 3) {
    const int64_t m = stack.at(1).toInt();
    meta.dtype = stack_tensor(stack, 2).scalar_type();
    meta.shape = {n, m};
  } else {
    meta.dtype = stack_tensor(stack, 1).scalar_type();
    meta.shape = {n, n};
  }
  return {meta};
}

SharedMetaDataVector EyeSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& tensor = stack_tensor(stack, stack.size() == 3 ? 2 : 1);
  const auto dtype = tensor.scalar_type();

  SharedMetaData constantSharedMeta{"constant"};
  constantSharedMeta.outputs_data.emplace_back(2, dtype);

  SharedMetaData matrixDiagSharedMeta("matrix_diagonal_fwd");
  matrixDiagSharedMeta.inputs_data = {{2, dtype}};
  matrixDiagSharedMeta.outputs_data = {{2, dtype}};
  return {constantSharedMeta, matrixDiagSharedMeta};
}

void EyeOpOut::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  std::vector<synapse_helpers::tensor> eye_out;
  auto meta = EyeMeta(stack)[0];

  auto computeDtype = meta.dtype;
  c10::optional<int> finalResultIndex = 0;
  if (meta.dtype == c10::ScalarType::Long) {
    computeDtype = c10::ScalarType::Int;
    finalResultIndex = c10::nullopt;
  }

  auto constant = ConstantHelper(graph, 1.0f, computeDtype, meta.shape);
  eye_out = BuildOp(
      graph,
      get_guid_with_precision("matrix_diagonal_fwd", computeDtype),
      {constant.get()},
      {{meta.shape, computeDtype, finalResultIndex}});
  if (meta.dtype != computeDtype) {
    auto castNode = BuildCast(
        this, graph, eye_out[0].get(), meta.shape, computeDtype, meta.dtype, 0);
    syn_out(0) = std::move(castNode);
  } else {
    syn_out(0) = std::move(eye_out[0]);
  }
}
} // namespace habana

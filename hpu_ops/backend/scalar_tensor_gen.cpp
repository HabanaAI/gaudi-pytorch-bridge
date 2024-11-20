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

#include "generated/backend/scalar_tensor.h"

namespace habana {

OutputMetaDataVector ScalarTensorMeta(const at::Stack& stack) {
  OutputMetaData meta;

  meta.dtype = stack.at(1).isNone() ? at::kFloat : stack.at(1).toScalarType();
  meta.layout = stack.at(2).isNone() ? at::kStrided : stack.at(2).toLayout();

  meta.shape = {}; // This should be a 0 dim tensor

  return {meta};
}

SharedMetaDataVector ScalarTensorSharedMeta(
    const at::Stack&,
    habana_helpers::HabanaExecutionMode) {
  // [SW-205149] return empty vector because shape tensor validation will block
  // shape agnostic flow
  return {};
}

void ScalarTensor::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto scalar = stack.at(0).toScalar();
  auto meta = ScalarTensorMeta(stack)[0];
  auto result = ConstantHelper(graph, scalar, meta.dtype, meta.shape, 0);
  syn_out(0) = std::move(result);
}
} // namespace habana

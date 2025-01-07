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

#include "generated/backend/fill.h"

namespace habana {
SharedMetaDataVector FillScalarSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& input = stack_tensor(stack, 0);
  const auto dtype = input.scalar_type();
  const auto rank = input.dim();
  SharedMetaTensor inOutTensor{rank, dtype};
  if (rank > 1) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data = {inOutTensor};
    return {constantSharedMeta};
  } else {
    SharedMetaData memcpySharedMeta{"memcpy"};
    memcpySharedMeta.inputs_data = {inOutTensor};
    memcpySharedMeta.outputs_data = {inOutTensor};
    return {memcpySharedMeta};
  }
}

void FillScalar::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto other = stack.at(1).toScalar();

  // If self is a ZST then return it as it is since there is nothing to fill
  if (!self.numel()) {
    const auto& outshape = stack_tensor(stack, 0).sizes();
    auto copy =
        BuildOp(graph, "memcpy", {syn_in(0)}, {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(copy[0]);
  } else {
    const auto& outshape = self.sizes();
    auto result = ConstantHelper(graph, other, ScalarType(), outshape, 0);
    syn_out(0) = std::move(result);
  }
}
} // namespace habana

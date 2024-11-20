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

#include "generated/backend/zero.h"

namespace habana {

SharedMetaDataVector ZeroSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  const auto rank = self.dim();
  if (rank > 1) {
    const auto dtype = self.scalar_type();
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data.emplace_back(rank, dtype);
    return {constantSharedMeta};
  }
  return {};
}

void ZeroHpuLazyOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  const auto outshape = self.sizes();
  const auto type = self.scalar_type();
  const auto value = 0;
  const auto final_output_index = 0;
  syn_out(0) = ConstantHelper(graph, value, type, outshape, final_output_index);
}

} // namespace habana

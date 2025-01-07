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

#include "generated/backend/_assert_async.h"

namespace habana {
SharedMetaDataVector AssertAsyncSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto self = stack_tensor(stack, 0);

  SharedMetaData assertAsyncSharedMeta{"assert_async"};
  assertAsyncSharedMeta.inputs_data.emplace_back(
      self.dim(), self.scalar_type());
  assertAsyncSharedMeta.outputs_data.emplace_back(1, c10::ScalarType::UInt32);
  return {assertAsyncSharedMeta};
}

void AssertAsync::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack.at(0).toTensor();

  std::string name = graph.name();
  std::size_t found = name.find("_");
  std::string substring = name.substr(found + 1);
  uint64_t graph_index = std::stoi(substring);
  synAssertAsyncParams params;
  params.msg_id = graph_index;

  auto assert_op =
      BuildOp(graph, "assert_async", {syn_in(0)}, {}, &params, sizeof(params));
}
} // namespace habana

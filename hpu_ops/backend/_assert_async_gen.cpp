/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************
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

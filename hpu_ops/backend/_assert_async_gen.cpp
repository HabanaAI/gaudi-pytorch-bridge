/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

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
#include "hpu_ops/hpu_op_helper.h"

namespace habana {
void AssertAsync::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  size_t size;

  std::string name = graph.name();
  std::size_t found = name.find("_");
  std::string substring = name.substr(found + 1);
  auto graph_index = std::stoi(substring);
  PARAMS_STUB(ns_AssertAsync::Params);
  params->node_id = static_cast<unsigned int>(graph_index);
  params->msg_id = 44;
  auto assert_op =
      BuildOp(graph, "assert_async", {syn_in(0)}, {}, &params, size);
}
} // namespace habana

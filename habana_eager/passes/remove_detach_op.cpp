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

#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/ir/ir.h>
#include "habana_eager/graph_exec.h"
#include "habana_helpers/logging_pt.h"

namespace habana {
namespace graph {
namespace pass {

bool RemoveDetachOp(std::shared_ptr<torch::jit::Graph> graph) {
  PT_EAGER_TRACE;
  auto nodes = graph->nodes();

  std::unordered_set<torch::jit::Node*> detach_nodes;
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    auto node = *it;
    if (node->kind() != torch::jit::aten::detach) {
      continue;
    }

    detach_nodes.insert(node);
  }

  for (auto n : detach_nodes) {
    n->output(0)->replaceAllUsesWith(n->input(0));
    n->destroy();
  }

  if (!detach_nodes.empty()) {
    PT_EAGER_INFO(__PRETTY_FUNCTION__, ": \n", *graph);
  }

  return !detach_nodes.empty();
}

} // namespace pass
} // namespace graph
} // namespace habana

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

#include "habana_eager/graph_exec.h"
#include "habana_helpers/logging_pt.h"

namespace habana {
namespace graph {
namespace pass {

void SanitizeGraphInput(std::shared_ptr<torch::jit::Graph> graph) {
  PT_EAGER_TRACE;
  if (0 == graph->inputs().size()) {
    // No input to sanitize...
    return;
  }

  torch::jit::Value* first_graph_input{*graph->inputs().begin()};
  if (!first_graph_input->hasUses() &&
      "self" == first_graph_input->debugName()) {
    graph->eraseInput(0);
  }
}

} // namespace pass
} // namespace graph
} // namespace habana
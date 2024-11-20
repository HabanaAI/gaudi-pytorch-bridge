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

// #include <c10/util/ArrayRef.h>

#include <cstddef>
#include <cstdint>
#include <queue>

#include "habana_eager/graph_exec.h"

#include "habana_eager/eager_view.h"
#include "habana_helpers/logging_pt.h"

namespace habana {
namespace graph {
namespace pass {

uint32_t const_id = 0;

struct MarkParamsAsConstPass {
  explicit MarkParamsAsConstPass(std::shared_ptr<torch::jit::Graph> graph)
      : m_graph(std::move(graph)) {}

  bool run(torch::jit::Stack& example_inputs) {
    PT_EAGER_TRACE;
    HABANA_ASSERT(m_graph->inputs().size() == example_inputs.size());
    auto index = 0;
    bool changed = false;
    for (auto input : m_graph->inputs()) {
      auto input_name = input->debugName();
      if (input_name.find("_frozen_param") != std::string::npos) {
        PT_EAGER_DEBUG("Frozen_param: ", input_name);
        if (example_inputs[index].isTensor()) {
          auto tensor = example_inputs[index].toTensor();
          auto set_const_id = habana::get_tensor_const_id(tensor);
          if (set_const_id == INVALID_CONST_ID) {
            habana::set_tensor_const(tensor, true, const_id);
            const_id++;
          } else {
            habana::set_tensor_const(tensor, true, set_const_id);
          }
          TensorExtraMeta::set_const_tensor(tensor, true);
          changed = true;
        }
      }
      index++;
    }
    PT_EAGER_DEBUG(
        "Freezing of parameters is enabled, num frozen params found: ",
        const_id);
    auto num_const_expected = GET_ENV_FLAG_NEW(PT_HPU_CHECK_NUM_CONSTS);
    if (num_const_expected > 0) {
      HABANA_ASSERT(
          const_id == num_const_expected,
          "num_const_expected: ",
          num_const_expected,
          " num_const_marked: ",
          const_id);
    }
    return changed;
  }

 private:
  std::shared_ptr<torch::jit::Graph> m_graph;
};

bool MarkParamsAsConst(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& example_inputs) {
  PT_EAGER_TRACE;
  MarkParamsAsConstPass pass{graph};
  bool changed{pass.run(example_inputs)};
  if (changed) {
    PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
  }
  return changed;
}

} // namespace pass
} // namespace graph
} // namespace habana
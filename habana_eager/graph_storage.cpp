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

#include "habana_eager/graph_storage.h"
#include "habana_eager/eager_context.h"

#include "habana_helpers/logging.h"

namespace habana {
namespace graph {

GraphStorage& GraphStorage::get() {
  static GraphStorage storage;
  return storage;
}

size_t GraphStorage::add_new_recipe(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& example_inputs,
    bool dynamic,
    bool inference,
    bool has_preallocated_outputs,
    bool has_randoms,
    InputSymbolIndexMap& in_symbol_idx_map,
    std::vector<habana_helpers::RangeInfo>& range_infos,
    bool mark_dynamic) {
  PT_EAGER_TRACE;
  habana::eager::JoinPendingPipelineThreads();
  size_t output_recipe_group_id{m_storage_vec.size()};
  m_storage_vec.emplace_back(
      output_recipe_group_id,
      graph,
      example_inputs,
      dynamic,
      inference,
      has_preallocated_outputs,
      has_randoms,
      in_symbol_idx_map,
      range_infos,
      mark_dynamic);
  PT_EAGER_DEBUG(
      "Recipe group added to storage. recipe_group_id: ",
      output_recipe_group_id);
  return output_recipe_group_id;
}

torch::jit::Stack GraphStorage::launch_recipe(
    size_t recipe_id,
    torch::jit::Stack& inputs,
    std::vector<at::Tensor>& outputs) {
  PT_EAGER_TRACE;
  PT_EAGER_DEBUG("Launching from recipe_group_id: ", recipe_id);
  HABANA_ASSERT(recipe_id < m_storage_vec.size());
  GraphExecsGroup& gexec = m_storage_vec.at(recipe_id);
  return gexec.launch(inputs, outputs);
}

void GraphStorage::reset_seeds() {
  PT_EAGER_TRACE;
  for (auto& g : m_storage_vec) {
    g.ResetSeed();
  }
}

} // namespace graph
} // namespace habana

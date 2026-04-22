/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include <string>

#include "backend/jit_graph_cache.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/graph_storage.h"
#include "habana_helpers/logging.h"

namespace habana::graph {

GraphStorage& GraphStorage::get() {
  static GraphStorage storage;
  return storage;
}

size_t GraphStorage::add_new_recipe(
    std::shared_ptr<habana_torch::jit::Graph> graph,
    const std::string& parent_graph_name,
    torch::jit::Stack& example_inputs,
    const std::vector<bool>& is_reusable,
    bool dynamic,
    bool inference,
    bool has_preallocated_outputs,
    bool has_randoms,
    InputSymbolIndexMap& in_symbol_idx_map,
    std::vector<habana_helpers::RangeInfo>& range_infos,
    std::vector<int64_t>& const_indexes,
    bool has_dynamic_marked_tensors) {
  PT_EAGER_TRACE;
  size_t output_recipe_group_id{m_storage_vec.size()};
  m_storage_vec.emplace_back(
      output_recipe_group_id,
      graph,
      parent_graph_name,
      example_inputs,
      is_reusable,
      dynamic,
      inference,
      has_preallocated_outputs,
      has_randoms,
      in_symbol_idx_map,
      range_infos,
      const_indexes,
      has_dynamic_marked_tensors);
  PT_EAGER_DEBUG(
      "Recipe group added to storage. recipe_group_id: ",
      output_recipe_group_id);
  return output_recipe_group_id;
}

torch::jit::Stack GraphStorage::launch_recipe(
    size_t recipe_id,
    torch::jit::Stack& inputs,
    std::vector<at::Tensor>& outputs,
    std::string& parent_graph_name) {
  PT_EAGER_TRACE;
  auto lowering_queue_length =
      HPUDeviceContext::lowering_thread().get_active_task_count();
  // The following code ensures consistency in graph names when capturing events
  // for profiler.cpp. In graph_exec.cpp, graph names are formatted like
  // graph_0009_fused_0_jit_0100000.
  // However, since the recipe id is not available here, we maintain uniformity
  // by considering the name up to "_jit", resulting in a format like
  // graph_0009_fused_0_jit.
  if (parent_graph_name.find("_fx") != std::string::npos) {
    parent_graph_name =
        parent_graph_name.replace(parent_graph_name.find("fx"), 2, "jit");
  }
  LOP::emit_event_fast(
      true,
      "LaunchRecipeTask()",
      parent_graph_name,
      LOP::PipelineStageID::PIPELINE_STAGE_LOWERING_ID,
      lowering_queue_length);
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

} // namespace habana::graph

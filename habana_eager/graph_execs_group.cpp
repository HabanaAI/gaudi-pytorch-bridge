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
// clang-format off
#include "backend/habana_device/HPUStream.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/jit_graph_cache.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/graph_execs_group.h"
#include "habana_eager/eager_tensor.h"
#include "habana_eager/graph_dynamic.h"
#include "habana_eager/graph_exec_passes.h"
#include "habana_eager/graph_storage.h"
#include "habana_eager/graph_weight_permute.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/thread_pool/thread_pool.h"

#include "habana_eager/eager_view.h"
// clang-format on

namespace habana {
namespace graph {

std::size_t GraphExecsGroup::generate_key(torch::jit::Stack& stack) {
  std::size_t rval = 0;

  for (const auto& input : stack) {
    if (!input.isTensor()) {
      continue;
    }
    torch::Tensor input_tensor{input.toTensor()};
    auto offset = input_tensor.storage_offset();

    rval = at::hash_combine(rval, offset);
  }

  return rval;
}

void GraphExecsGroup::RunGraphGroupPasses() {
  pass::SanitizeGraphInput(m_original_graph);
}

size_t GraphExecsGroup::generate_graph_index() {
  static const size_t graph_index_prefix = 1'000'000;
  return graph_index_prefix + m_graph_group_index * 1'000 +
      m_graph_exec_storage.size();
}

void GraphExecsGroup::CopyGraphAndEmplace(
    size_t key,
    torch::jit::Stack& stack) {
  auto graph_copy = m_original_graph->copy();

  m_graph_exec_storage.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(key),
      std::forward_as_tuple(
          generate_graph_index(),
          graph_copy,
          stack,
          m_dynamic,
          m_inference,
          m_has_preallocated_outputs,
          m_has_randoms,
          m_in_symbol_idx_map,
          m_range_infos,
          m_mark_dynamic));
}

GraphExecsGroup::GraphExecsGroup(
    size_t recipe_id,
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& example_inputs,
    bool dynamic,
    bool inference,
    bool has_preallocated_outputs,
    bool has_randoms,
    InputSymbolIndexMap in_symbol_idx_map,
    std::vector<habana_helpers::RangeInfo>& range_infos,
    bool mark_dynamic)
    : m_graph_group_index(recipe_id),
      m_original_graph(graph),
      m_dynamic(dynamic),
      m_inference(inference),
      m_has_preallocated_outputs(has_preallocated_outputs),
      m_has_randoms(has_randoms),
      m_in_symbol_idx_map(in_symbol_idx_map),
      m_range_infos(range_infos),
      m_mark_dynamic(mark_dynamic) {
  PT_EAGER_TRACE;

  m_graphs_group_name = "graphs_group_" + std::to_string(recipe_id);

  RunGraphGroupPasses();

  auto key = generate_key(example_inputs);
  CopyGraphAndEmplace(key, example_inputs);
}

torch::jit::Stack GraphExecsGroup::launch(
    torch::jit::Stack& stack,
    std::vector<at::Tensor>& outputs) {
  PT_EAGER_TRACE_WITH_NAME(m_graphs_group_name);

  auto key = generate_key(stack);

  if (m_graph_exec_storage.count(key) == 0) {
    PT_EAGER_DEBUG(
        "Not found proper subversion of " + m_graphs_group_name +
        ", create new flavor");

    CopyGraphAndEmplace(key, stack);
  }

  return m_graph_exec_storage.at(key).launch(stack, outputs);
}

void GraphExecsGroup::ResetSeed() {
  for (auto& pair : m_graph_exec_storage) {
    pair.second.ResetSeed();
  }
}

} // namespace graph
} // namespace habana

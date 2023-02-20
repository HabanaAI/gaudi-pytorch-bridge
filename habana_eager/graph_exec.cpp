/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "habana_eager/graph_exec.h"
#include "backend/jit_graph_cache.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "habana_device/HPUStream.h"
#include "habana_helpers/logging.h"

namespace habana {
namespace graph {

GraphStorage& GraphStorage::get() {
  static GraphStorage storage;
  return storage;
}

size_t GraphStorage::add_new_recipe(
    std::shared_ptr<torch::jit::Graph> graph,
    std::vector<at::Tensor>& example_inputs,
    bool dynamic,
    bool inference) {
  size_t output_recipe_id{m_storage_vec.size()};
  m_storage_vec.emplace_back(output_recipe_id, graph, dynamic, inference);
  return output_recipe_id;
}

std::vector<at::Tensor> GraphStorage::launch_recipe(
    size_t recipe_id,
    std::vector<at::Tensor>& inputs) {
  HABANA_ASSERT(recipe_id < m_storage_vec.size());
  return m_storage_vec[recipe_id].launch(inputs);
}

GraphExec::GraphExec(
    size_t recipe_id,
    std::shared_ptr<torch::jit::Graph> graph,
    bool dynamic,
    bool inference)
    : m_graph_index(recipe_id),
      m_graph(graph),
      m_dynamic(dynamic),
      m_inference(inference) {
  m_graph_name = "habana_graph_" + std::to_string(m_graph_index);
  // Remove 'self' from graph parameters
  m_graph->eraseInput(0);
  PT_EAGER_DEBUG("Compling graph (recipe_id: ", recipe_id, ")\n", *m_graph);
}

std::vector<at::Tensor> GraphExec::launch(std::vector<at::Tensor>& inputs) {
  PT_EAGER_TRACE;

  const c10::hpu::HPUStream& stream{c10::hpu::getCurrentHPUStream()};
  synEventHandle event_handle{};
  synapse_helpers::hpuStream_t event_stream{0};
  bool event_flag{0};

  torch::jit::Stack stack;
  stack.reserve(inputs.size());

  for (const auto& in : inputs) {
    stack.emplace_back(in);
  }

  at::ArrayRef<torch::jit::IValue> input_refs =
      torch::jit::last(stack, m_graph->inputs().size());

  auto graph_and_meta{std::make_shared<habana::OptimizedJITGraphAndMetaData>(
      m_graph,
      input_refs,
      0ull /*unique_cntr*/,
      std::vector<bool>{} /*node_bcast_map_*/)};

  graph_and_meta->SetGraphIndex(m_graph_index);
  graph_and_meta->SetOpName(m_graph_name);
  graph_and_meta->SetHPUStream(stream);
  graph_and_meta->SetEventHandle(event_handle);
  graph_and_meta->SetEventRecordStream(event_stream);
  graph_and_meta->SetEventFlag(event_flag);

  try {
    habana::HabanaLaunchOpPT habana_launch_op_{graph_and_meta};
    habana_launch_op_.run(stack);

    std::vector<at::Tensor> outputs;
    size_t num_outputs{m_graph->outputs().size()};
    outputs.reserve(num_outputs);

    for (size_t ii = 0; ii < num_outputs; ii++) {
      HABANA_ASSERT(stack[ii].isTensor());
      outputs.push_back(stack[ii].toTensor());
    }
    return outputs;
  } catch (const std::exception& e) {
    PT_EAGER_FATAL("HabanaLaunchOpPT Run returned exception....\n", e.what());
  }
}

} // namespace graph
} // namespace habana

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
#include "backend/habana_device/HPUStream.h"
#include "backend/jit_graph_cache.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/graph_weight_permute.h"

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
    bool inference) {
  size_t output_recipe_id{m_storage_vec.size()};
  m_storage_vec.emplace_back(
      output_recipe_id, graph, example_inputs, dynamic, inference);
  return output_recipe_id;
}

torch::jit::Stack GraphStorage::launch_recipe(
    size_t recipe_id,
    torch::jit::Stack& inputs) {
  HABANA_ASSERT(recipe_id < m_storage_vec.size());
  return m_storage_vec[recipe_id].launch(inputs);
}

GraphExec::GraphExec(
    size_t recipe_id,
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& example_inputs,
    bool dynamic,
    bool inference)
    : m_graph_index(recipe_id),
      m_graph(graph),
      m_dynamic(dynamic),
      m_inference(inference) {
  PT_EAGER_TRACE;

  m_graph_name = "habana_graph_" + std::to_string(m_graph_index);

  RunGraphPasses();
  PT_EAGER_INFO("Jit for ", m_graph_name, ":\n", *m_graph);

  at::ArrayRef<torch::jit::IValue> input_refs =
      torch::jit::last(example_inputs, m_graph->inputs().size());

  m_graph_and_meta = std::make_shared<habana::OptimizedJITGraphAndMetaData>(
      m_graph,
      input_refs,
      0ull /*unique_cntr*/,
      std::vector<bool>{} /*node_bcast_map_*/);

  m_graph_and_meta->SetGraphIndex(m_graph_index);
  m_graph_and_meta->SetOpName(m_graph_name);
};

void GraphExec::RunGraphPasses() {
  PT_EAGER_DEBUG("Jit for ", m_graph_name, " before passes\n", *m_graph);
  pass::SanitizeGraphInput(m_graph);
  pass::DetectWeightTensors(m_graph, m_graph_inputs_to_permute);
  pass::ReplaceGetItemWithListUnpack(m_graph);
  pass::HandleTupleOnOutput(m_graph);
  pass::AddAttributeAlpha(m_graph);
}

torch::jit::Stack GraphExec::launch(torch::jit::Stack& stack) {
  PT_EAGER_TRACE;
  habana::eager::SingleTonEagerContext::getInstance()
      .JoinPendingLoweringThread();

  const c10::hpu::HPUStream& stream{c10::hpu::getCurrentHPUStream()};

  at::ArrayRef<torch::jit::IValue> input_refs =
      torch::jit::last(stack, m_graph->inputs().size());

  HandleWeightPermutation(stack);

  m_graph_and_meta->SetHPUStream(stream);

  try {
    habana::HabanaLaunchOpPT habana_launch_op_{m_graph_and_meta};
    habana_launch_op_.run(stack);
    return stack;
  } catch (const std::exception& e) {
    PT_EAGER_FATAL("HabanaLaunchOpPT Run returned exception....\n", e.what());
  }
}

void GraphExec::HandleWeightPermutation(torch::jit::Stack& stack) {
  for (auto input : m_graph_inputs_to_permute) {
    c10::IValue input_value{stack[input]};
    HABANA_ASSERT(input_value.isTensor());
    torch::Tensor weight_tensor{input_value.toTensor()};
    habana::graph::PermuteWeightTensor t(weight_tensor);
    t.PermuteIfNeeded();
  }
}

} // namespace graph
} // namespace habana

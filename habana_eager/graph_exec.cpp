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
#include "habana_eager/eager_view.h"
#include "habana_eager/graph_weight_permute.h"

#include "habana_helpers/logging.h"

namespace habana {
namespace graph {

size_t generate_graph_index(size_t recipe_id) {
  static const size_t graph_index_prefix = 100000;
  return graph_index_prefix + recipe_id;
}

GraphStorage& GraphStorage::get() {
  static GraphStorage storage;
  return storage;
}

size_t GraphStorage::add_new_recipe(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& example_inputs,
    bool dynamic,
    bool inference) {
  PT_EAGER_TRACE;
  habana::eager::SingleTonEagerContext::getInstance()
      .JoinPendingLoweringThread();

  size_t output_recipe_id{m_storage_vec.size()};
  m_storage_vec.emplace_back(
      output_recipe_id, graph, example_inputs, dynamic, inference);
  PT_EAGER_DEBUG("Recipe added to storage. recipe_id: ", output_recipe_id);
  return output_recipe_id;
}

torch::jit::Stack GraphStorage::launch_recipe(
    size_t recipe_id,
    torch::jit::Stack& inputs) {
  PT_EAGER_TRACE;
  PT_EAGER_DEBUG("Launching recipe_id: ", recipe_id);
  habana::eager::SingleTonEagerContext::getInstance()
      .JoinPendingLoweringThread();
  HABANA_ASSERT(recipe_id < m_storage_vec.size());
  return m_storage_vec[recipe_id].launch(inputs);
}

GraphExec::GraphExec(
    size_t recipe_id,
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& example_inputs,
    bool dynamic,
    bool inference)
    : m_graph_index(generate_graph_index(recipe_id)),
      m_graph(graph),
      m_dynamic(dynamic),
      m_inference(inference) {
  PT_EAGER_TRACE;

  m_graph_name = "graph_recipe_" + std::to_string(recipe_id);

  RunGraphPasses(example_inputs);
  LogRecipeInfo(example_inputs);

  at::ArrayRef<torch::jit::IValue> input_refs =
      torch::jit::last(example_inputs, m_graph->inputs().size());

  m_graph_and_meta = std::make_shared<habana::OptimizedJITGraphAndMetaData>(
      m_graph,
      input_refs,
      0ull /*unique_cntr*/,
      std::vector<bool>{} /*node_bcast_map_*/,
      m_graph_name);

  m_graph_and_meta->SetGraphIndex(m_graph_index);
  m_graph_and_meta->SetOpName(m_graph_name);
  m_graph_and_meta->set_is_eager_compiler_supported(false);
};

void GraphExec::LogRecipeInfo(torch::jit::Stack& example_inputs) {
  PT_EAGER_INFO("Jit for ", m_graph_name, ":\n", *m_graph);

  for (int input_idx = 0; input_idx < m_graph->inputs().size(); input_idx++) {
    std::string perm = "";
    if (example_inputs[input_idx].isTensor()) {
      torch::Tensor tensor{example_inputs[input_idx].toTensor()};
      synapse_helpers::layouts::MemoryPermutation m_perm;
      std::tie(m_perm, std::ignore) =
          habana_helpers::get_tensor_memory_permutation(tensor);
      std::string m_perm_s(m_perm.begin(), m_perm.end());
      perm = " Perm: " + m_perm_s;
    }

    PT_EAGER_INFO(
        m_graph->inputs().at(input_idx)->debugName(),
        ": ",
        habana_helpers::DebugString(example_inputs[input_idx]),
        perm);
  }
}

void GraphExec::RunGraphPasses(torch::jit::Stack& example_inputs) {
  PT_EAGER_TRACE;
  PT_EAGER_DEBUG("Jit for ", m_graph_name, " before passes\n", *m_graph);
  pass::SanitizeGraphInput(m_graph);
  pass::DetectWeightTensors(m_graph, m_graph_inputs_to_permute);
  pass::HandleInputViews(m_graph, example_inputs, m_input_new_base_sizes);
  pass::ReplaceGetItemWithListUnpack(m_graph);
  pass::HandleTupleOnOutput(m_graph);
  pass::AddAttributeAlpha(m_graph);
  pass::RemoveDetachOp(m_graph);
}

torch::jit::Stack GraphExec::launch(torch::jit::Stack& original_stack) {
  PT_EAGER_TRACE_WITH_NAME(m_graph_name);
  torch::jit::Stack stack =
      habana::eager::convert_inputs_to_backend_tensors(original_stack);

  const c10::hpu::HPUStream& stream{c10::hpu::getCurrentHPUStream()};

  at::ArrayRef<torch::jit::IValue> input_refs =
      torch::jit::last(stack, m_graph->inputs().size());

  HandleWeightPermutation(stack);

  for (auto& input_base_sizes_pair : m_input_new_base_sizes) {
    int64_t input_idx{input_base_sizes_pair.first};
    std::vector<int64_t> base_sizes{input_base_sizes_pair.second};

    HABANA_ASSERT(input_refs.at(input_idx).isTensor());
    torch::Tensor input_tensor{input_refs.at(input_idx).toTensor()};

    auto* impl = input_tensor.unsafeGetTensorImpl();
    impl->set_storage_offset(0);
    impl->set_sizes_contiguous(base_sizes);
  }

  m_graph_and_meta->SetHPUStream(stream);

  try {
    habana::HabanaLaunchOpPT habana_launch_op_{m_graph_and_meta};
    habana_launch_op_.run(stack);
    return stack;
  } catch (const std::exception& e) {
    PT_EAGER_FATAL("HabanaLaunchOpPT Run returned exception....\n", e.what());
  }
} // namespace graph

void GraphExec::HandleWeightPermutation(torch::jit::Stack& stack) {
  PT_EAGER_TRACE;
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

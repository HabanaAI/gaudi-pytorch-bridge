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
#include "backend/synapse_helpers/env_flags.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/eager_tensor.h"
#include "habana_eager/eager_view.h"
#include "habana_eager/graph_dynamic.h"
#include "habana_eager/graph_exec_passes.h"
#include "habana_eager/graph_storage.h"
#include "habana_eager/graph_weight_permute.h"
#include "habana_eager/passes/handle_views_insert_permute.h"

#include "habana_helpers/logging.h"

namespace habana {
namespace graph {

size_t generate_graph_index(size_t recipe_id) {
  static const size_t graph_index_prefix = 100000;
  return graph_index_prefix + recipe_id;
}

void GraphExec::LaunchRecipeTask(
    GraphExec& gexec,
    torch::jit::Stack& inputs,
    std::vector<at::Tensor>& outputs) {
  PT_EAGER_TRACE_WITH_NAME(gexec.m_graph_name);
  try {
    gexec.LaunchRecipe(inputs, outputs);
  } catch (const std::exception& e) {
    PT_BRIDGE_WARN(
        "Exception caught in Lowering thread (will be rethrown in main thread)...\n",
        e.what());
    habana::eager::SingleTonEagerContext::getInstance()
        .StoreLoweringThreadException(std::current_exception());

  } catch (...) {
    PT_BRIDGE_WARN(
        "Exception caught in Lowering thread (will be rethrown in main thread)...\n");
    habana::eager::SingleTonEagerContext::getInstance()
        .StoreLoweringThreadException(std::current_exception());
  }
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

  habana::eager::JoinPendingPipelineThreads();

  m_graph_name = "graph_recipe_" + std::to_string(recipe_id);

  m_is_pipeline_supported =
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE) && !m_dynamic;

  RunGraphPasses(example_inputs);
  LogRecipeInfo(example_inputs);
  PT_DYNAMIC_SHAPE_DEBUG("Is Dynamic Graph = ", IsDynamicGraph());
  torch::jit::Stack in_stack = example_inputs;
  if (IsDynamicGraph()) {
    ProcessDynamicGraph(example_inputs);
    in_stack = ProcessDynamicStack(example_inputs, true);
  }

  at::ArrayRef<torch::jit::IValue> input_refs =
      torch::jit::last(in_stack, m_graph->inputs().size());

  std::string jit_graph_name = "";
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_JIT_GRAPH_NAME_HASH)) {
    jit_graph_name = m_graph_name;
  }

  m_graph_and_meta = std::make_shared<habana::OptimizedJITGraphAndMetaData>(
      m_graph,
      input_refs,
      0ull /*unique_cntr*/,
      std::vector<bool>{} /*node_bcast_map_*/,
      jit_graph_name,
      IsDynamicGraph());

  m_graph_and_meta->SetGraphIndex(m_graph_index);
  m_graph_and_meta->SetFrontendType(
      habana_helpers::HabanaFrontendTypes::COMPILE);
  m_graph_and_meta->SetOpName(m_graph_name);
  m_graph_and_meta->set_is_eager_compiler_supported(false);
};

bool GraphExec::IsDynamicGraph() {
  static bool is_refine_dynamic{
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)};
  return m_dynamic && is_refine_dynamic;
}

void GraphExec::ProcessDynamicGraph(torch::jit::Stack& example_inputs) {
  m_dgraph_meta = std::make_shared<DynamicGraphMetaData>();
  pass::HandleDynamicOps(m_graph, example_inputs, m_dgraph_meta);
  PT_EAGER_DEBUG(
      "Jit for ", m_graph_name, " after processing dynamicity\n", *m_graph);
}

std::vector<at::IValue> GraphExec::ProcessDynamicStack(
    torch::jit::Stack& orig_stack,
    bool is_first_launch) {
  PT_EAGER_TRACE;
  torch::jit::Stack new_stack;
  new_stack.reserve(
      orig_stack.size() + m_dgraph_meta->ds_input_patching_list.size());
  new_stack.insert(new_stack.end(), orig_stack.begin(), orig_stack.end());
  pass::HandleDynamicInputPatching(new_stack, m_dgraph_meta, is_first_launch);
  HABANA_ASSERT(
      m_graph->inputs().size() == new_stack.size(),
      "Graph inputs size not patching with stack size!!");
  if (!is_first_launch && m_dgraph_meta->negative_size_nodes.size())
    pass::ResolveNegativeSTSizes(m_graph, new_stack, m_dgraph_meta);
  return new_stack;
}

std::string GraphExec::LogRecipeInfo(torch::jit::Stack& example_inputs) {
  PT_EAGER_INFO(
      "Jit for ",
      m_graph_name,
      " dynamic: ",
      m_dynamic,
      " inference: ",
      m_inference,
      ":\n",
      *m_graph);

  for (int input_idx = 0; input_idx < m_graph->inputs().size(); input_idx++) {
    if (example_inputs[input_idx].isTensor()) {
      torch::Tensor tensor{example_inputs[input_idx].toTensor()};
      synapse_helpers::layouts::MemoryPermutation m_perm;
      std::tie(m_perm, std::ignore) =
          habana_helpers::get_tensor_memory_permutation(tensor);
      PT_EAGER_INFO(
          m_graph->inputs().at(input_idx)->debugName(),
          ": ",
          habana_helpers::DebugString(example_inputs[input_idx]),
          " Perm: ",
          VecToString(m_perm));
    }
  }

  return "";
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
  pass::HandleStridedViewsAndInsertPermute(m_graph);
  pass::GetOutputsOrderInGraph(m_graph, m_outputs_order);
}

torch::jit::Stack GraphExec::launch(
    torch::jit::Stack& stack,
    std::vector<at::Tensor>& outputs) {
  PT_EAGER_TRACE_WITH_NAME(m_graph_name);
  if (IsDynamicGraph()) {
    // For dynamic shapes we do not use preallocated outputs - value is returned
    // as stack. If python layer allocated outputs, it will probably disregard
    // our stack modification, and return this output instead - so it will
    // result with random/unitilized value.
    HABANA_ASSERT(outputs.size() == 0);

    return LaunchDynamicRecipe(stack);
  }

  torch::jit::Stack backend_inputs =
      habana::eager::convert_ivalues_to_backend_tensors(stack);

  std::vector<at::Tensor> backend_outputs;
  backend_outputs.reserve(outputs.size());
  for (auto& tensor : outputs) {
    backend_outputs.push_back(
        habana::eager::HbEagerTensorPool::getInstance().get_backend_tensor(
            tensor));
  }

  HandleWeightPermutation(backend_inputs);

  if (m_is_pipeline_supported && backend_outputs.size() > 0) {
    habana::eager::SingleTonEagerContext::getInstance()
        .ScheduleWorkAndUpdateLoweringThreadHandle(
            [this,
             backend_inputs = std::move(backend_inputs),
             backend_outputs = std::move(backend_outputs)]() mutable {
              return hpu_registrar().get_device().get_lowering_thread().enqueue(
                  LaunchRecipeTask, *this, backend_inputs, backend_outputs);
            });
    return {};
  } else {
    std::optional<std::vector<at::Tensor>> maybe_backend_outputs;
    if (backend_outputs.size() > 0) {
      maybe_backend_outputs = backend_outputs;
    }
    habana::eager::JoinPendingPipelineThreads();
    torch::jit::Stack ret_stack =
        LaunchRecipe(backend_inputs, maybe_backend_outputs);
    return habana::eager::convert_ivalues_to_backend_tensors(ret_stack);
  }
}

torch::jit::Stack GraphExec::LaunchDynamicRecipe(
    torch::jit::Stack& original_stack) {
  PT_EAGER_TRACE;
  PT_EAGER_INFO("LaunchDynamicRecipe. is_first_launch: ", is_first_launch);

  habana::eager::JoinPendingPipelineThreads();

  torch::jit::Stack stack =
      ProcessDynamicStack(original_stack, is_first_launch);
  is_first_launch = false;

  // [TODO] Disable hybrid sif until SW-153320
  habana_helpers::SetHybridSIFTorchCompile(false);

  PT_EAGER_INFO("Dynamic graph Info:", LogRecipeInfo(stack));

  torch::jit::Stack backend_inputs =
      habana::eager::convert_ivalues_to_backend_tensors(stack);

  HandleWeightPermutation(backend_inputs);

  torch::jit::Stack ret_stack = LaunchRecipe(backend_inputs);
  habana_helpers::SetHybridSIFTorchCompile(true);
  return habana::eager::convert_ivalues_to_backend_tensors(ret_stack);
}

torch::jit::Stack GraphExec::LaunchRecipe(
    torch::jit::Stack& stack,
    std::optional<std::vector<at::Tensor>> maybe_outputs) {
  // Important - this function is meant to be run on lowering thread.
  PT_EAGER_TRACE;

  if (maybe_outputs.has_value() && maybe_outputs.value().size() > 0) {
    std::vector<at::Tensor>& outputs{maybe_outputs.value()};
    std::vector<at::Tensor> reordered_outputs;
    HABANA_ASSERT(m_outputs_order.size() == outputs.size());
    reordered_outputs.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
      reordered_outputs.push_back(outputs[m_outputs_order[i]]);
    }
    maybe_outputs = reordered_outputs;
  }

  const c10::hpu::HPUStream& stream{c10::hpu::getCurrentHPUStream()};

  at::ArrayRef<torch::jit::IValue> input_refs =
      torch::jit::last(stack, m_graph->inputs().size());

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
    habana_launch_op_.run(stack, maybe_outputs);
    return stack;
  } catch (const std::exception& e) {
    PT_EAGER_FATAL("HabanaLaunchOpPT Run returned exception....\n", e.what());
  }
}

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

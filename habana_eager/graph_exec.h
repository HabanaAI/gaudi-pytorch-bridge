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

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <torch/csrc/jit/ir/ir.h>
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "habana_eager/graph_dynamic.h"

namespace habana {
namespace graph {

class GraphExec {
 public:
  GraphExec(
      size_t recipe_id,
      std::shared_ptr<torch::jit::Graph> graph,
      torch::jit::Stack& example_inputs,
      bool dynamic,
      bool inference,
      bool has_preallocated_outputs);

  torch::jit::Stack launch(
      torch::jit::Stack& inputs,
      std::vector<at::Tensor>& outputs);

  static void LaunchRecipeTask(
      GraphExec* gexec,
      torch::jit::Stack&& inputs,
      std::vector<at::Tensor>&& outputs);

  GraphExec() = delete;
  GraphExec(const GraphExec&) = delete;
  GraphExec(GraphExec&&) = default;
  GraphExec& operator=(const GraphExec&) = delete;

 private:
  torch::jit::Stack LaunchDynamicRecipe(torch::jit::Stack& inputs);
  torch::jit::Stack LaunchRecipe(
      torch::jit::Stack stack,
      std::optional<std::vector<at::Tensor>> maybe_outputs = {});

  void RunGraphPasses(torch::jit::Stack& example_inputs);
  std::string LogRecipeInfo(torch::jit::Stack& example_inputs);
  void HandleWeightPermutation(torch::jit::Stack& stack);
  bool IsDynamicGraph();
  void ProcessDynamicGraph(torch::jit::Stack& example_inputs);
  std::vector<c10::IValue> ProcessDynamicStack(torch::jit::Stack& stack, bool);

  size_t m_graph_index;
  std::shared_ptr<torch::jit::Graph> m_graph;
  std::string m_graph_name;
  bool m_dynamic;
  bool m_inference;
  bool is_first_launch = true;
  bool m_is_pipeline_supported = false;
  std::shared_ptr<DynamicGraphMetaData> m_dgraph_meta = nullptr;

  std::shared_ptr<habana::OptimizedJITGraphAndMetaData> m_graph_and_meta;
  std::set<int> m_graph_inputs_to_permute;
  std::map<int64_t, std::vector<int64_t>> m_input_new_base_sizes;
  std::vector<size_t> m_outputs_order;
  bool m_has_preallocated_outputs = false;
};

} // namespace graph
} // namespace habana

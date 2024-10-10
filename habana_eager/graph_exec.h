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
#include "habana_eager/graph_dynamic_ops.h"

namespace habana {
namespace graph {

using InputSymbolIndexMap = std::unordered_map<std::string, int64_t>;

class GraphExec {
 public:
  GraphExec(
      size_t recipe_id,
      std::shared_ptr<torch::jit::Graph> graph,
      torch::jit::Stack& example_inputs,
      bool dynamic,
      bool inference,
      bool has_preallocated_outputs,
      bool has_randoms,
      InputSymbolIndexMap in_symbol_idx_map,
      std::vector<habana_helpers::RangeInfo>& range_infos,
      bool mark_dynamic);

  torch::jit::Stack launch(
      torch::jit::Stack& inputs,
      std::vector<at::Tensor>& outputs);

  static void LaunchRecipeTask(
      GraphExec* gexec,
      torch::jit::Stack&& inputs,
      std::vector<at::Tensor>&& outputs,
      LaunchDynamicShapes launch_shapes,
      InputSymbolMap&& in_symbol_value_map);

  void ResetSeed();

  GraphExec() = delete;
  GraphExec(const GraphExec&) = delete;
  GraphExec(GraphExec&&) = default;
  GraphExec& operator=(const GraphExec&) = delete;

 private:
  torch::jit::Stack LaunchDynamicRecipe(torch::jit::Stack& inputs);
  torch::jit::Stack LaunchRecipe(
      torch::jit::Stack stack,
      std::optional<std::vector<at::Tensor>> maybe_outputs = {},
      InputSymbolMap in_symbol_value_map = {});

  void RunGraphPasses(torch::jit::Stack& example_inputs);
  void RunPass(
      std::function<bool()> pass,
      bool dump_graphs,
      const std::string& pass_name);
  std::string LogRecipeInfo(torch::jit::Stack& example_inputs);
  bool IsDynamicGraph();
  void ProcessDynamicGraph(torch::jit::Stack& example_inputs);
  std::vector<c10::IValue> ProcessDynamicStack(torch::jit::Stack& stack, bool);
  void UpdateSeedTensors(torch::jit::Stack& stack);
  bool HasInvalidDynanmicSymbols();

  struct SeedTensors {
    std::optional<at::Tensor> seed;
    std::optional<at::Tensor> counter;
  };

  size_t m_graph_index;
  std::shared_ptr<torch::jit::Graph> m_graph;
  std::string m_graph_name;
  bool m_dynamic;

  bool m_static_fallback = false;
  bool m_inference;
  bool is_first_launch = true;
  bool m_is_pipeline_supported = false;
  std::shared_ptr<DynamicGraphMetaData> m_dgraph_meta = nullptr;
  DynamicPatchingData m_ds_patch_data;

  std::shared_ptr<habana::OptimizedJITGraphAndMetaData> m_graph_and_meta;
  std::set<int> m_graph_inputs_to_permute;
  std::map<int64_t, std::vector<int64_t>> m_input_new_base_sizes;
  std::vector<size_t> m_outputs_order;
  bool m_has_preallocated_outputs = false;
  const bool m_has_randoms;
  InputSymbolIndexMap m_in_symbol_idx_map;
  std::vector<habana_helpers::RangeInfo> m_range_infos;
  bool m_mark_dynamic = false;
  bool m_reset_seed = true;
  SeedTensors m_seed_tensors{};
  size_t m_sym_expr_hash = 0;
};

} // namespace graph
} // namespace habana

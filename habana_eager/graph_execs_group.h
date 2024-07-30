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
#include <unordered_map>
#include <vector>

#include <torch/csrc/jit/ir/ir.h>
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "habana_eager/graph_dynamic.h"
#include "habana_eager/graph_exec.h"

namespace habana {
namespace graph {

using InputSymbolIndexMap = std::unordered_map<std::string, int64_t>;

struct GraphExecsGroup {
  GraphExecsGroup(
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

  void ResetSeed();

  GraphExecsGroup() = delete;
  GraphExecsGroup(const GraphExecsGroup&) = delete;
  GraphExecsGroup(GraphExecsGroup&&) = default;
  GraphExecsGroup& operator=(const GraphExecsGroup&) = delete;

 private:
  size_t m_graph_group_index;
  std::shared_ptr<torch::jit::Graph> m_original_graph;
  std::string m_graphs_group_name;
  bool m_dynamic;
  bool m_inference;

  bool m_has_preallocated_outputs = false;
  const bool m_has_randoms;
  InputSymbolIndexMap m_in_symbol_idx_map;
  std::vector<habana_helpers::RangeInfo> m_range_infos;
  bool m_mark_dynamic = false;

  std::unordered_map<int, GraphExec> m_graph_exec_storage;

  std::size_t generate_key(torch::jit::Stack& stack);

  void CopyGraphAndEmplace(size_t key, torch::jit::Stack& stack);

  size_t generate_graph_index();

  void RunGraphGroupPasses();
};

} // namespace graph
} // namespace habana

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

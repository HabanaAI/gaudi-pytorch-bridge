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

#include <vector>

#include <torch/csrc/jit/ir/ir.h>
#include "habana_eager/graph_execs_group.h"

namespace habana {
namespace graph {

class GraphStorage {
 public:
  static GraphStorage& get();

  size_t add_new_recipe(
      std::shared_ptr<torch::jit::Graph> graph,
      torch::jit::Stack& example_inputs,
      bool dynamic,
      bool inference,
      bool has_preallocated_outputs,
      bool has_randoms,
      InputSymbolIndexMap& in_symbol_idx_map,
      std::vector<habana_helpers::RangeInfo>& range_infos,
      bool mark_dynamic);
  torch::jit::Stack launch_recipe(
      size_t recipe_id,
      torch::jit::Stack& inputs,
      std::vector<at::Tensor>& outputs);
  void reset_seeds();

 private:
  GraphStorage(){};
  GraphStorage(const GraphStorage&) = delete;
  GraphStorage& operator=(const GraphStorage&) = delete;
  GraphStorage(GraphStorage&&) = delete;
  GraphStorage& operator=(GraphStorage&&) = delete;

  std::vector<GraphExecsGroup> m_storage_vec;
};

} // namespace graph
} // namespace habana

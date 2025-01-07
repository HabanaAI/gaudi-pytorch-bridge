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

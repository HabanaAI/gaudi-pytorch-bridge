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

#include <c10/util/ArrayRef.h>

#include <map>
#include <queue>

#include "habana_eager/graph_exec.h"
#include "habana_helpers/logging_pt.h"

namespace habana {
namespace graph {
namespace pass {

struct DetectWeightTensorsPass {
  explicit DetectWeightTensorsPass(std::shared_ptr<torch::jit::Graph> graph)
      : m_graph(std::move(graph)) {}

  bool run() {
    processInputs(m_graph->inputs());
    return false;
  }

  std::set<int> get_weight_input_indices() {
    return m_weight_input_indices;
  }

 private:
  void processInputs(at::ArrayRef<torch::jit::Value*> inputs) {
    for (size_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
      torch::jit::Value* input{inputs.at(input_idx)};
      for (auto& use : input->uses()) {
        bool is_weight_input{processInputUse(input, use)};
        if (is_weight_input) {
          m_weight_input_indices.insert(input_idx);
          m_weight_inputs.insert(input);
          break;
        }
      }
    }
  }

  bool processInputUse(torch::jit::Value* input, const torch::jit::Use& use) {
    torch::jit::Node* user{use.user};
    return isWeightInput(input, user);
  }

  bool isWeightInput(
      torch::jit::Value* initial_input,
      torch::jit::Node* user_node) {
    std::queue<std::pair<torch::jit::Value*, torch::jit::Node*>> nodes_to_visit;

    nodes_to_visit.push(std::make_pair(initial_input, user_node));

    while (!nodes_to_visit.empty()) {
      auto input_node_pair{nodes_to_visit.front()};
      torch::jit::Value* input{input_node_pair.first};
      torch::jit::Node* node{input_node_pair.second};

      if (isDirectConvoWeightInput(input, node)) {
        // No point for further graph searching
        PT_EAGER_INFO(
            "Convolution weight input detected: ", input->debugName());
        return true;
      }

      static const c10::Symbol cast_symbol{
          c10::Symbol::fromQualString("aten::to")};
      static const c10::Symbol cast_copy_symbol{
          c10::Symbol::fromQualString("aten::_to_copy")};
      static const int cast_tensor_input_idx{0};
      if ((cast_symbol == node->kind()) || (cast_copy_symbol == node->kind())) {
        HABANA_ASSERT(node->outputs().size() == 1);
        torch::jit::Value* cast_input{node->input(cast_tensor_input_idx)};
        torch::jit::Value* cast_output{node->output(0)};
        // Node looks like valid cast so we need to look for nodes that using
        // it's output
        if (cast_input == input)
          for (auto& use : cast_output->uses()) {
            torch::jit::Node* user_node{use.user};
            nodes_to_visit.push(std::make_pair(cast_output, user_node));
          }
      }
      nodes_to_visit.pop();
    }

    return false;
  }

  bool isDirectConvoWeightInput(
      torch::jit::Value* input,
      torch::jit::Node* user_node) {
    static const std::map<c10::Symbol, size_t> conv_symbols_map{
        {c10::Symbol::fromQualString("aten::convolution"), 1},
        {c10::Symbol::fromQualString("aten::convolution_backward"), 2},
        {c10::Symbol::fromQualString("aten::convolution_overrideable"), 1},
        {c10::Symbol::fromQualString("aten::convolution_backward_overrideable"),
         2}};

    if (conv_symbols_map.find(user_node->kind()) != conv_symbols_map.end()) {
      const size_t weight_input_idx{conv_symbols_map.at(user_node->kind())};
      HABANA_ASSERT(user_node->inputs().size() >= weight_input_idx);
      torch::jit::Value* weight_input{user_node->input(weight_input_idx)};

      if (input == weight_input) {
        return true;
      }
    }
    return false;
  }

  std::set<int> m_weight_input_indices;
  std::set<torch::jit::Value*> m_weight_inputs;
  std::shared_ptr<torch::jit::Graph> m_graph;
}; // namespace pass

void DetectWeightTensors(
    std::shared_ptr<torch::jit::Graph> graph,
    std::set<int>& indices_to_permute) {
  PT_EAGER_TRACE;
  DetectWeightTensorsPass pass{graph};
  pass.run();
  indices_to_permute = pass.get_weight_input_indices();
}

} // namespace pass
} // namespace graph
} // namespace habana

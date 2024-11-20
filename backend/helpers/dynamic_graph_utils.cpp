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

#include "backend/helpers/dynamic_graph_utils.h"

namespace habana_helpers {

bool is_symbolic_expr(const std::string& expr_str) {
  for (auto& c : expr_str) {
    if (!(std::isdigit(c) || c == '[' || c == ']' || c == ',' ||
          std::isspace(c)))
      return true;
  }
  return false;
}

bool is_output_shape_empty(const std::string& expr_str) {
  // expr_str is ""
  if (expr_str.empty())
    return true;
  // expr_str is of the form "[[]]" or "[[], []]" ...
  for (auto& c : expr_str) {
    if (!(c == '[' || c == ']' || c == ','))
      return false;
  }
  return true;
}

bool nodeHasScalarGraphInput(
    torch::jit::Node* node,
    GraphInputIndexMap& org_stack_index_map,
    CValuePtrToIValuePtrMap& value_ivalue_map) {
  for (const auto& input : node->inputs()) {
    torch::jit::Node* producer_node = input->node();
    if (producer_node->kind() == torch::jit::prim::ListConstruct)
      return nodeHasScalarGraphInput(
          producer_node, org_stack_index_map, value_ivalue_map);
    else {
      auto ivalue = value_ivalue_map[const_cast<torch::jit::Value*>(input)];
      if (!ivalue->isTensor()) {
        if (org_stack_index_map.count(input->debugName())) {
          auto node_name = node->kind().toQualString();
          PT_EAGER_DEBUG(
              "Node ",
              node_name,
              " has scalar inputs that are also graph inputs");
          return true;
        }
      }
    }
  }
  return false;
}

bool isNodeDynamic(
    torch::jit::Node* node,
    GraphInputIndexMap& org_stack_index_map,
    CValuePtrToIValuePtrMap& value_ivalue_map) {
  // Assuming node is dynamic by default
  bool isDynamic = true;
  auto node_name = node->kind().toQualString();
  auto outputshapes_attr = c10::Symbol::attr("output_shapes");
  if (node->hasAttribute(outputshapes_attr)) {
    auto outputshapes_str = node->s(outputshapes_attr);
    if (is_output_shape_empty(outputshapes_str)) {
      PT_EAGER_DEBUG(
          "output_shapes attr is empty for node = ",
          node_name,
          ", assuming it to be dynamic");
    } else {
      bool hasSymbol = is_symbolic_expr(outputshapes_str);
      // Node is not dynamic if it does not have
      // any non-numeric symbols
      if (!hasSymbol)
        isDynamic = false;
      // If node has scalar inputs that are also graph inputs
      // Differing values of those inputs cause JIT cache miss
      // Better to replace such nodes
      if (nodeHasScalarGraphInput(node, org_stack_index_map, value_ivalue_map))
        isDynamic = true;
    }
  } else {
    PT_EAGER_DEBUG(
        "output_shapes attr is missing for node = ",
        node_name,
        ", assuming it to be dynamic");
  }
  return isDynamic;
}

void createGraphInputStackIndexMap(
    const std::shared_ptr<torch::jit::Graph>& graph,
    GraphInputIndexMap& org_stack_index_map) {
  for (size_t idx = 0; idx < graph->inputs().size(); ++idx) {
    auto input = graph->inputs().at(idx);
    auto name = input->debugName();
    org_stack_index_map[name] = idx;
  }
}

size_t CalculateSymbolValuesHash(InputSymbolMap& symbol_value_map) {
  size_t hash_code = 0;
  for (const auto& pair : symbol_value_map) {
    auto symbol_hash = c10::get_hash(pair.first);
    hash_code = c10::hash_combine(hash_code, symbol_hash);
    const double* value_ptr = pair.second.get();
    auto value_hash = value_ptr ? c10::get_hash(*value_ptr) : DBL_MAX;
    hash_code = c10::hash_combine(hash_code, value_hash);
  }
  return hash_code;
}

} // namespace habana_helpers
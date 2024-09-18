/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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

} // namespace habana_helpers
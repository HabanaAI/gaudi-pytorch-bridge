/**
 * Copyright (c) 2021-2026 Intel Corporation
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
#include "replace_inplace_ops.h"
#include <torch/csrc/jit/ir/irparser.h>
#include "habana_helpers/logging.h"

namespace habana_lazy {

static const std::unordered_map<std::string, std::string> inPlaceToOutOfPlace =
    {
        {"aten::add_", "aten::add"},
        {"hpu::add_", "hpu::add"},
        {"aten::div_", "aten::div"},
        {"aten::index_put_", "aten::index_put"},
        {"aten::mul_", "aten::mul"},
        {"aten::relu_", "aten::relu"},
        {"aten::leaky_relu_", "aten::leaky_relu"},
        {"aten::clamp_", "aten::clamp"},
        {"aten::sub_", "aten::sub"},
        // Idemponent transformation inplace -> inplace, it is quick fix for
        // invalid detection of graph inputs in some cases.
        {"aten::zero_", "aten::zero_"},
        {"aten::index_copy_", "aten::index_copy"},
};

bool isInplaceOp(const torch::jit::Node* node) {
  return (node != nullptr)
      ? inPlaceToOutOfPlace.count(node->kind().toQualString()) != 0
      : false;
}

bool isControlNode(const torch::jit::Node* node) {
  return (node != nullptr)
      ? (node->kind().toQualString() == std::string("hpu::control_edge_"))
      : false;
}

bool isInList(
    const std::vector<torch::jit::Value*>& l,
    const torch::jit::Value* v) {
  return std::find(l.begin(), l.end(), v) != l.end();
}

bool checkOps(const torch::jit::Node* n) {
  return isInplaceOp(n) || isControlNode(n);
}

bool isGraphInput(
    const std::shared_ptr<torch::jit::Graph>& graph,
    const torch::jit::Value* v) {
  auto inputs = graph->inputs().vec();
  if (isInList(inputs, v)) {
    return true;
  }

  const auto* n = v->node();
  if ((n != nullptr) && !n->inputs().empty()) {
    auto* in = n->input(0);
    if (checkOps(n) && isGraphInput(graph, in)) {
      return true;
    }
  }

  return false;
}

bool isGraphOutput(
    const std::shared_ptr<torch::jit::Graph>& graph,
    const torch::jit::Value* v) {
  auto outputs = graph->outputs().vec();
  if (isInList(outputs, v)) {
    return true;
  }

  for (const auto& u : v->uses()) {
    auto* n = u.user;
    if ((n != nullptr) && checkOps(n) && !n->outputs().empty()) {
      auto* o = n->output(0);
      if (isGraphOutput(graph, o)) {
        return true;
      }
    }
  }

  return false;
}

/*
 * A Inplace op can be replaced if the below conditions
 * are met:
 * (Handle only single output node)
 * 1. Node output is not part of graph output
 * 2. Node input is not part of graph input
 *
 * Relaxed check for replacing in-place ops:

If below conditions are satisfied, then check for graph output is avoided.
1. Node not connected to input
 */
bool canReplaceOp(
    const std::shared_ptr<torch::jit::Graph>& graph,
    const torch::jit::Node* node) {
  if ((nullptr == node) || (node->outputs().size() > 1) ||
      node->inputs().empty()) {
    return false;
  }

  auto* in = node->input(0);
  return (isInplaceOp(node) && !isGraphInput(graph, in));
}

void replace_inplace_ops(
    std::shared_ptr<torch::jit::Graph>& graph,
    const std::vector<torch::jit::Node*>& nodes) {
  for (const auto& node : nodes) {
    if (nullptr == node) {
      continue;
    }
    torch::jit::WithInsertPoint insert_point(node);

    std::string kind = node->kind().toQualString();
    const std::string& new_kind = inPlaceToOutOfPlace.at(kind);

    auto* new_node = graph->create(c10::Symbol::fromQualString(new_kind));
    new_node->addInput(node->input(0));
    for (size_t i = 1; i < node->inputs().size(); ++i) {
      new_node->addInput(node->input(i));
    }
    new_node->setScope(node->scope());
    new_node->copyAttributes(*node);
    new_node->output(0)->copyMetadata(node->output(0));
    graph->insertNode(new_node);
    node->output(0)->replaceAllUsesWith(new_node->output(0));
    node->destroy();
  }
}

void replace_inplace_ops(std::shared_ptr<torch::jit::Graph>& graph) {
  std::vector<torch::jit::Node*> inplace_ops;

  for (auto* node : graph->nodes()) {
    if (canReplaceOp(graph, node)) {
      inplace_ops.emplace_back(node);
    }
  }

  replace_inplace_ops(graph, inplace_ops);
}

}; // namespace habana_lazy

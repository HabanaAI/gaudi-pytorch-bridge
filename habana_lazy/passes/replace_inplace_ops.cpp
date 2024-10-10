/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "replace_inplace_ops.h"
#include <torch/csrc/jit/ir/irparser.h>
#include "habana_helpers/logging.h"

namespace habana_lazy {

using Graph = torch::jit::Graph;
using Value = torch::jit::Value;
using Node = torch::jit::Node;

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

bool isInplaceOp(const Node* node) {
  return node ? inPlaceToOutOfPlace.count(node->kind().toQualString()) != 0
              : false;
}

bool isControlNode(const Node* node) {
  return node
      ? ((node->kind().toQualString() == std::string("hpu::control_edge_")) ||
         (node->kind().toQualString() ==
          std::string("hpu::control_edge_other_")))
      : false;
}

bool isInList(const std::vector<Value*>& l, const Value* v) {
  return std::find(l.begin(), l.end(), v) != l.end();
}

bool checkOps(const Node* n) {
  std::string kind = n->kind().toQualString();
  return isInplaceOp(n) || isControlNode(n);
}

bool isGraphInput(const std::shared_ptr<Graph>& graph, const Value* v) {
  auto inputs = graph->inputs().vec();
  if (isInList(inputs, v)) {
    return true;
  }

  auto n = v->node();
  if (n && (n->inputs().size() >= 1)) {
    auto in = n->input(0);
    if (checkOps(n) && isGraphInput(graph, in)) {
      return true;
    }
  }

  return false;
}

bool isGraphOutput(const std::shared_ptr<Graph>& graph, const Value* v) {
  auto outputs = graph->outputs().vec();
  if (isInList(outputs, v)) {
    return true;
  }

  for (auto& u : v->uses()) {
    auto n = u.user;
    if (n && checkOps(n) && (n->outputs().size() >= 1)) {
      auto o = n->output(0);
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
bool canReplaceOp(const std::shared_ptr<Graph>& graph, const Node* node) {
  if ((nullptr == node) || (node->outputs().size() > 1) ||
      (node->inputs().size() < 1)) {
    return false;
  }

  auto in = node->input(0);
  if (isInplaceOp(node) && !isGraphInput(graph, in)) {
    return true;
  }
  return false;
}

void replace_inplace_ops(
    std::shared_ptr<Graph>& graph,
    const std::vector<Node*>& nodes) {
  for (auto& node : nodes) {
    torch::jit::WithInsertPoint insert_point(node);
    if (nullptr == node) {
      continue;
    }

    std::string kind = node->kind().toQualString();
    std::string new_kind = inPlaceToOutOfPlace.at(kind);

    auto new_node = graph->create(c10::Symbol::fromQualString(new_kind));
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

void replace_inplace_ops(std::shared_ptr<Graph>& graph) {
  std::vector<Node*> inplace_ops;

  for (auto node : graph->nodes()) {
    if (canReplaceOp(graph, node)) {
      inplace_ops.emplace_back(node);
    }
  }

  replace_inplace_ops(graph, inplace_ops);
}

}; // namespace habana_lazy

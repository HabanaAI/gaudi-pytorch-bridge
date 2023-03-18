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
#include <string>
#include <vector>

#include <torch/csrc/jit/ir/ir.h>

namespace habana {
namespace graph {

namespace pass {
void SanitizeGraphInput(std::shared_ptr<torch::jit::Graph>& graph);
void HandleTupleOnOutput(std::shared_ptr<torch::jit::Graph>& graph);
} // namespace pass

class GraphExec {
 public:
  GraphExec(
      size_t recipe_id,
      std::shared_ptr<torch::jit::Graph> graph,
      bool dynamic,
      bool inference);

  torch::jit::Stack launch(torch::jit::Stack& inputs);

 private:
  size_t m_graph_index;
  std::shared_ptr<torch::jit::Graph> m_graph;
  std::string m_graph_name;
  bool m_dynamic;
  bool m_inference;
};

class GraphStorage {
 public:
  static GraphStorage& get();

  size_t add_new_recipe(
      std::shared_ptr<torch::jit::Graph> graph,
      torch::jit::Stack& example_inputs,
      bool dynamic,
      bool inference);
  torch::jit::Stack launch_recipe(size_t recipe_id, torch::jit::Stack& inputs);

 private:
  GraphStorage(){};
  GraphStorage(const GraphStorage&) = delete;
  GraphStorage& operator=(const GraphStorage&) = delete;
  GraphStorage(GraphStorage&&) = delete;
  GraphStorage& operator=(GraphStorage&&) = delete;

  std::vector<habana::graph::GraphExec> m_storage_vec;
};

} // namespace graph
} // namespace habana
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

#include <pybind11/stl.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/extension.h>
#include "backend/helpers/tensor_utils.h"
#include "habana_eager/graph_storage.h"

#include "habana_helpers/logging.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "graph_compile",
      [](std::shared_ptr<torch::jit::Graph> graph,
         const py::tuple& inputs,
         bool dynamic,
         bool inference) {
        torch::jit::Stack stack;
        stack.reserve(inputs.size());
        for (auto& obj : inputs) {
          stack.push_back(torch::jit::toTypeInferredIValue(obj));
        }
        auto& graph_storage{habana::graph::GraphStorage::get()};
        return graph_storage.add_new_recipe(graph, stack, dynamic, inference);
      },
      py::return_value_policy::copy,
      py::arg("graph"),
      py::arg("inputs"),
      py::arg("dynamic"),
      py::arg("inference"));
  m.def(
      "graph_launch",
      [](size_t recipe_id,
         const py::tuple& inputs,
         std::vector<at::Tensor>& outputs) {
        torch::jit::Stack stack;
        stack.reserve(inputs.size());
        for (auto& obj : inputs) {
          stack.push_back(torch::jit::toTypeInferredIValue(obj));
        }

        auto& graph_storage{habana::graph::GraphStorage::get()};
        stack = graph_storage.launch_recipe(recipe_id, stack, outputs);

        if (outputs.size() == 0) {
          return torch::jit::createPyObjectForStack(std::move(stack));
        }

        torch::jit::Stack out_stack;
        for (size_t idx = 0; idx < outputs.size(); idx++) {
          out_stack.push_back(outputs[idx]);
        }
        return torch::jit::createPyObjectForStack(std::move(out_stack));
      },
      py::return_value_policy::copy,
      py::arg("recipe_id"),
      py::arg("inputs"),
      py::arg("outputs"));
}
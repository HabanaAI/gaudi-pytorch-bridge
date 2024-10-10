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
namespace {

using InputSymbolIndexMap = std::unordered_map<std::string, int64_t>;
struct EmptyBatchData {
  std::vector<int64_t> size;
  py::object dtype;
  std::optional<std::vector<int64_t>> stride;
  EmptyBatchData(
      std::vector<int64_t> size,
      py::object dtype,
      std::optional<std::vector<int64_t>> stride)
      : size(std::move(size)), dtype(dtype), stride(std::move(stride)) {}
};

std::vector<at::Tensor> batch_empty(const std::vector<EmptyBatchData>& batch) {
  auto allocator = habana::getHABANADeviceAllocator();
  constexpr c10::DispatchKeySet hpu_ks(c10::DispatchKey::HPU);

  std::vector<at::Tensor> result;
  for (const auto& el : batch) {
    at::ScalarType dtype_c =
        reinterpret_cast<THPDtype*>(el.dtype.ptr())->scalar_type;
    auto dtype = dtype_or_default(dtype_c);
    HABANA_ASSERT(habana_helpers::is_supported_type(dtype));

    if (!el.stride.has_value()) {
      result.push_back(
          at::detail::empty_generic(el.size, allocator, hpu_ks, dtype, {}));
    } else {
      result.push_back(at::detail::empty_strided_generic(
          el.size, el.stride.value(), allocator, hpu_ks, dtype));
    }
  }
  return result;
}

std::size_t calculate_hash_code(const py::tuple& inputs) {
  torch::jit::Stack stack;
  stack.reserve(inputs.size());
  for (auto& obj : inputs)
    stack.push_back(torch::jit::toTypeInferredIValue(obj));

  size_t hash_code = 0;
  for (const auto& input : stack) {
    if (!input.isTensor())
      continue;
    const auto& tensor = input.toTensor();
    if (!tensor.defined())
      continue;
    size_t tensor_hash = c10::get_hash(tensor.sizes(), tensor.strides());
    hash_code = c10::hash_combine(hash_code, tensor_hash);
  }
  return hash_code;
}
}; // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "graph_compile",
      [](std::shared_ptr<torch::jit::Graph> graph,
         const py::tuple& inputs,
         bool dynamic,
         bool inference,
         bool has_preallocated_outputs,
         bool has_randoms,
         InputSymbolIndexMap& in_symbol_idx_map,
         std::vector<habana_helpers::RangeInfo>& range_infos,
         bool mark_dynamic) {
        torch::jit::Stack stack;
        stack.reserve(inputs.size());
        for (auto& obj : inputs) {
          stack.push_back(torch::jit::toTypeInferredIValue(obj));
        }
        auto& graph_storage{habana::graph::GraphStorage::get()};
        return graph_storage.add_new_recipe(
            graph,
            stack,
            dynamic,
            inference,
            has_preallocated_outputs,
            has_randoms,
            in_symbol_idx_map,
            range_infos,
            mark_dynamic);
      },
      py::return_value_policy::copy,
      py::arg("graph"),
      py::arg("inputs"),
      py::arg("dynamic"),
      py::arg("inference"),
      py::arg("has_preallocated_outputs"),
      py::arg("has_randoms"),
      py::arg("in_symbol_idx_map"),
      py::arg("range_infos"),
      py::arg("mark_dynamic"));
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
  m.def("reset_seeds", []() {
    auto& graph_storage{habana::graph::GraphStorage::get()};
    graph_storage.reset_seeds();
  });
  py::class_<EmptyBatchData>(m, "EmptyBatchData")
      .def(py::init<
           std::vector<int64_t>,
           py::object,
           std::optional<std::vector<int64_t>>>())
      .def_readwrite("size", &EmptyBatchData::size);
  m.def("batch_empty", &batch_empty, "Create empty tensors");
  py::class_<habana_helpers::RangeInfo>(m, "RangeInfo")
      .def(py::init<
           std::vector<int64_t>,
           std::vector<int64_t>,
           std::string,
           std::string,
           int>())
      .def_readwrite("min_shape", &habana_helpers::RangeInfo::min_shape)
      .def_readwrite("max_shape", &habana_helpers::RangeInfo::max_shape)
      .def_readwrite("expr", &habana_helpers::RangeInfo::expr)
      .def_readwrite("expr_strides", &habana_helpers::RangeInfo::expr_strides)
      .def_readwrite("index", &habana_helpers::RangeInfo::index);
  m.def(
      "calculate_hash_code",
      &calculate_hash_code,
      "Calculate hash key of graph input tensor shapes and strides");
}

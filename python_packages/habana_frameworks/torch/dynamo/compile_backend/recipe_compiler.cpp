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

#include <pybind11/stl.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/extension.h>
#include "backend/habana_device/HPUAllocator.h"
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

std::size_t calculate_symval_hashcode(
    const py::tuple& inputs,
    InputSymbolIndexMap& symbol_idx_map) {
  size_t hash_code = 0;
  std::for_each(
      symbol_idx_map.begin(),
      symbol_idx_map.end(),
      [&](const std::pair<std::string, int64_t>& p) {
        int64_t scalar_index = p.second;
        HABANA_ASSERT(
            scalar_index >= 0 ||
                static_cast<size_t>(scalar_index) < inputs.size(),
            "Symbol index received is out of bounds!!",
            scalar_index);
        const auto& obj = inputs[scalar_index];
        HABANA_ASSERT(
            py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj),
            "Expected scalar but got non-scalar object!!",
            scalar_index);
        double value;
        if (py::isinstance<py::float_>(obj)) {
          value = obj.cast<double>(); // Cast directly to double if it's a float
        } else {
          value = static_cast<double>(
              obj.cast<int64_t>()); // Cast to double if it's an int
        }

        auto symbol_hash = c10::get_hash(p.first);
        hash_code = c10::hash_combine(hash_code, symbol_hash);
        auto value_hash = c10::get_hash(value);
        hash_code = c10::hash_combine(hash_code, value_hash);
      });
  return hash_code;
}
}; // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "graph_compile",
      [](std::shared_ptr<torch::jit::Graph> graph,
         const py::tuple& inputs,
         const py::tuple& is_reusable,
         bool dynamic,
         bool inference,
         bool has_preallocated_outputs,
         bool has_randoms,
         InputSymbolIndexMap& in_symbol_idx_map,
         std::vector<habana_helpers::RangeInfo>& range_infos,
         std::vector<int64_t>& const_indexes,
         bool mark_dynamic) {
        torch::jit::Stack stack;
        stack.reserve(inputs.size());
        for (auto& obj : inputs) {
          stack.push_back(torch::jit::toTypeInferredIValue(obj));
        }
        std::vector<bool> is_reusable_vec;
        is_reusable_vec.reserve(is_reusable.size());
        for (auto& obj : is_reusable) {
          is_reusable_vec.push_back(obj.cast<bool>());
        }
        auto& graph_storage{habana::graph::GraphStorage::get()};
        return graph_storage.add_new_recipe(
            graph,
            stack,
            is_reusable_vec,
            dynamic,
            inference,
            has_preallocated_outputs,
            has_randoms,
            in_symbol_idx_map,
            range_infos,
            const_indexes,
            mark_dynamic);
      },
      py::return_value_policy::copy,
      py::arg("graph"),
      py::arg("inputs"),
      py::arg("is_reusable"),
      py::arg("dynamic"),
      py::arg("inference"),
      py::arg("has_preallocated_outputs"),
      py::arg("has_randoms"),
      py::arg("in_symbol_idx_map"),
      py::arg("range_infos"),
      py::arg("const_indexes"),
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
      "calculate_symval_hashcode",
      &calculate_symval_hashcode,
      "Calculate hash key of graph input tensor shapes and strides");
}

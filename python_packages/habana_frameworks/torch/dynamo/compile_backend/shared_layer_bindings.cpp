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

#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "op_def.h"

namespace py = pybind11;

bool check_cpu_fallback_op(
    std::string op,
    c10::FunctionSchema schema,
    bool allow_numbers_as_tensors,
    bool is_dynamic,
    const py::list& shared_meta,
    py::args args,
    const py::kwargs& kwargs) {
  if (hpu_shared_layer_unsupported_ops.find(op) !=
      hpu_shared_layer_unsupported_ops.end()) {
    return false;
  }
  if (fallback_support_check_map.find(op) != fallback_support_check_map.end()) {
    bool check_kernel_support = fallback_support_check_map[op](
        schema,
        allow_numbers_as_tensors,
        is_dynamic,
        shared_meta,
        args,
        kwargs);
    return not check_kernel_support;
  }
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("check_cpu_fallback_op", &check_cpu_fallback_op, "CPU fallback check");
}

/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

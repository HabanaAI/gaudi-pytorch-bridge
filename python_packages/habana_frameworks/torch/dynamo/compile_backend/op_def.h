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

#pragma once

#include <ATen/core/function_schema.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;
extern std::unordered_map<
    std::string,
    std::function<bool(
        c10::FunctionSchema&,
        bool,
        bool,
        const py::list&,
        py::args&,
        const py::dict&)>>
    fallback_support_check_map;

extern std::set<std::string> hpu_shared_layer_unsupported_ops;

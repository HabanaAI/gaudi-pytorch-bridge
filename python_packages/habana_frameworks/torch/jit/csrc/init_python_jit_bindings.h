/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "pybind11/stl.h"

namespace habana_torch {
namespace jit {
TORCH_API void InitBindings(pybind11::module& m);
} // namespace jit
} // namespace habana_torch

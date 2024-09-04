/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "jit_fork/ir/ir.h"

namespace habana_torch {
namespace jit {

TORCH_API TypePtr getTensorType(const at::Tensor& t, bool complete);

TORCH_API TypePtr inferShapeAndTypeForInput(
    TypePtr input_type,
    torch::jit::Stack::const_iterator& s_iter,
    const torch::jit::Stack::const_iterator& s_iter_end,
    bool complete);

TORCH_API void setInputTensorTypes(
    Graph& g,
    const torch::jit::Stack& stack,
    bool complete,
    const std::vector<int>& param_count_list = {});

} // namespace jit
} // namespace habana_torch

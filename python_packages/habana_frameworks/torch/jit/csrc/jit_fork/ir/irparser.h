/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>

#include <c10/util/Optional.h>

#include <torch/csrc/Export.h>

namespace habana_torch {
namespace jit {

struct Graph;
struct Value;

// \brief Parse IR from \p STR constructing the corresponding IR in\ GRAPH.
// if parse_tensor_constants is true will construct empty tensors
// for Tensor constants with random or unitialized contents, otherwise will
// throw
TORCH_API void parseIR(
    const std::string& str,
    Graph* graph,
    bool parse_tensor_constants = false);

/** \brief Parse IR from \p STR constructing the corresponding IR in\ GRAPH.
 *
 * \p VMAP is filled with String to Value pairs allowing to index Values in the
 * newly created graph by their name in the original IR string.
 * if parse_tensor_constants is true will construct empty tensors
 * for Tensor constants with random or unitialized contents, otherwise will
 * throw
 */
TORCH_API void parseIR(
    const std::string& str,
    Graph* graph,
    std::unordered_map<std::string, Value*>& vmap,
    bool parse_tensor_constants = false);

} // namespace jit
} // namespace habana_torch

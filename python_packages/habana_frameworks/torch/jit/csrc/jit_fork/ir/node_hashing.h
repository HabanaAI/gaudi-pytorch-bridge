/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "jit_fork/ir/ir.h"

namespace habana_torch {
namespace jit {

struct TORCH_API HashNode {
  size_t operator()(const Node* k) const;
};

struct TORCH_API EqualNode {
  bool operator()(const Node* lhs, const Node* rhs) const;
};

} // namespace jit
} // namespace habana_torch

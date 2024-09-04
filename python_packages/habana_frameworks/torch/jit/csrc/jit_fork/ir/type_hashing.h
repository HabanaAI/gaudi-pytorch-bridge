/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/core/jit_type.h>

#include "jit_fork/ir/ir.h"

namespace habana_torch {
namespace jit {

struct HashType {
  size_t operator()(const TypePtr& type) const;
  size_t operator()(const c10::ConstTypePtr& type) const;
};

struct EqualType {
  bool operator()(const TypePtr& a, const TypePtr& b) const;
  bool operator()(const c10::ConstTypePtr& a, const c10::ConstTypePtr& b) const;
};

} // namespace jit
} // namespace habana_torch

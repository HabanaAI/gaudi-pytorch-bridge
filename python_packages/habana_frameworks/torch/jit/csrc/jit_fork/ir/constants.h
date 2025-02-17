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
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>

#include <torch/csrc/Export.h>

#include "jit_fork/frontend/source_range.h"
#include "jit_fork/ir/scope.h"

// helpers for handling constants in the IR
// - create constant nodes from ints, floats, complex, intlist, Tensors, and
// other types
// - implement primitive constant ops.
namespace habana_torch {
namespace jit {

using ::c10::IValue;

struct Graph;
struct Value;

// thrown when insertConstant cannot encode the IValue into a graph
struct TORCH_API constant_not_supported_error : public std::runtime_error {
  using runtime_error::runtime_error;
};

TORCH_API Value* insertConstant(
    Graph& g,
    const IValue& val,
    c10::optional<SourceRange> loc = c10::nullopt,
    c10::optional<ScopePtr> scope = c10::nullopt);

// note: prefer g.insertConsant(val, loc) which does exactly the same thing
// this function is only declared/defined here because its implementation is
// closely related to the implementation of prim::Constant that is also in
// constants.cpp.
//
// returns a c10::nullopt if the IValue kind cannot be inserted as a constant
TORCH_API c10::optional<Value*> tryInsertConstant(
    Graph& g,
    const IValue& val,
    c10::optional<SourceRange> loc = c10::nullopt,
    c10::optional<ScopePtr> scope = c10::nullopt);

////////////////////////////////////////////////////////////////////////////////
// Helper for retrieving constants
////////////////////////////////////////////////////////////////////////////////

// attempt to convert a (possibly constant) Value* into an interpreter value
// (IValue). returns c10::nullopt if the Value* was not constant
TORCH_API c10::optional<IValue> toIValue(const Value* v);

// if a value is a constant then try to turn into type T using the
// same rules as the interpreter
template <typename T>
c10::optional<T> constant_as(const Value* v) {
  if (auto ivalue = toIValue(v)) {
    return ivalue->to<T>();
  }
  return c10::nullopt;
}

} // namespace jit
} // namespace habana_torch

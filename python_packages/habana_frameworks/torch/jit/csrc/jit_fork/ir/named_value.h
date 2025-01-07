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

#include <type_traits>

#include <ATen/core/ivalue.h>

#include <torch/csrc/utils/variadic.h>

#include "habana_helpers/logging.h"
#include "jit_fork/frontend/source_range.h"
#include "jit_fork/ir/constants.h"

namespace habana_torch {
namespace jit {

struct Value;

/**
 * A value with optional extra name and location information. Used during
 * schema matching to provide extra error information and resolve kwargs.
 */
struct NamedValue {
  NamedValue(const SourceRange& loc, const std::string& name, Value* value)
      : loc_(loc), name_(name), value_(value) {}
  NamedValue(const SourceRange& loc, Value* value) : loc_(loc), value_(value) {}

  /* implicit */ NamedValue(Value* value) : value_(value) {}
  NamedValue(const std::string& name, Value* value)
      : name_(name), value_(value) {}

  /* implicit */ NamedValue(IValue value)
      : value_(nullptr), ivalue_(std::move(value)) {}

  NamedValue(const std::string& name, IValue value)
      : name_(name), ivalue_(std::move(value)) {}

  template <
      typename T,
      typename = std::enable_if_t<
          (!std::is_same_v<std::decay_t<T>, NamedValue> &&
           !std::is_same_v<std::decay_t<T>, Value*> &&
           !std::is_same_v<std::decay_t<T>, IValue>)>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  NamedValue(T&& t) : NamedValue(IValue(std::forward<T>(t))) {}

  template <
      typename T,
      typename = std::enable_if_t<
          (!std::is_same_v<std::decay_t<T>, Value*> &&
           !std::is_same_v<std::decay_t<T>, IValue>)>>
  NamedValue(const std::string& name, T&& t)
      : NamedValue(name, IValue(std::forward<T>(t))) {}

  SourceRange locOr(const SourceRange& backup_location) const {
    if (!loc_)
      return backup_location;
    return loc();
  }

  // note: this will insert a constant node into the graph at the current
  // insert point if this NamedValue is actually a constant
  Value* value(Graph& g) const {
    if (!value_)
      return insertConstant(
          g, ivalue_); // use insertConstant to remove need to include ir.h here
    return value_;
  }

  const std::string& name() const {
    HABANA_ASSERT(name_);
    return *name_;
  }

  const SourceRange& loc() const {
    HABANA_ASSERT(loc_);
    return *loc_;
  }

  at::TypePtr type() const;

 private:
  c10::optional<SourceRange> loc_;
  c10::optional<std::string> name_;
  Value* value_{nullptr};
  // only valid if value_ == nullptr;
  IValue ivalue_;
};

} // namespace jit
} // namespace habana_torch

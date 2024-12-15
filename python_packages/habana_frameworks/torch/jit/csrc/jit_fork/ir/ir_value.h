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

#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/dynamic_type.h>
#include <ATen/core/enum_type.h>
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>

#include "jit_fork/ir/type_wrapper.h"

using c10::TensorType;

namespace habana_torch::jit {

using ::c10::TypePtr;
using use_list = std::vector<Use>;

struct Node;

struct Value {
  AT_DISALLOW_COPY_AND_ASSIGN(Value);
  Value(Node* node_, size_t offset_);

 private:
  friend struct Node;
  friend struct Graph;
  Node* node_;
  size_t offset_;
  size_t unique_ = 0; // unique id
  use_list uses_;
  std::string unique_name_;
  TypeWrapper type_wrapper_;
  // a managing wrapper for Python to allow invalidation
  std::shared_ptr<Wrap<Value>> wrap_;

 public:
  // ::torch::jit::Value* getUpstreamValue(::torch::jit::Value*);
  Value* setType(TypePtr type);
  Value* setType(const TypeWrapper& type_wrapper);
  TORCH_API void inferTypeFrom(const at::Tensor& output);
  TORCH_API void inferTypeFrom(
      const c10::intrusive_ptr<c10::ivalue::Object>& output);
  const TypePtr& type() const {
    return type_wrapper_.getType();
  }
  const TypeWrapper& typeWrapper() const {
    return type_wrapper_;
  }
  bool requires_grad() const {
    return type()->requires_grad();
  }
  bool isCompleteTensor() const {
    if (auto pt = type()->cast<TensorType>()) {
      return pt->isComplete();
    }
    return false;
  }
  TORCH_API bool mustBeNone() const;
  TORCH_API bool mustNotBeNone() const;
  size_t unique() const {
    return unique_;
  }
  bool hasDebugName() const {
    return !unique_name_.empty();
  }
  static bool isValidName(const std::string& name, bool allow_numbers = false);
  TORCH_API Value* setDebugName(
      const std::string& name,
      bool allow_numbers = false);
  std::string debugName() const {
    if (hasDebugName()) {
      return unique_name_;
    }
    return std::to_string(unique());
  }
  TORCH_API std::string debugNameBase() const;
  Node* node() {
    return node_;
  }
  size_t offset() const {
    return offset_;
  }
  void setOffset(size_t offset) {
    offset_ = offset;
  }
  const Node* node() const {
    return node_;
  }

  /**
   * @warning NEVER pass raw pointer of smart pointer managed Graph to Python.
   * Check #87343 for details.
   */
  Graph* owningGraph();
  const Graph* owningGraph() const;
  // TODO: make this more const correct
  const use_list& uses() const {
    return uses_;
  }

  bool hasUses() const {
    return !uses().empty();
  }

  TORCH_API void replaceFirstUseWith(Value* newValue);

  // Replaces all uses of this value with 'newValue'.
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%3, %3)
  // Execute: %3.replaceAllUsesWith(%6)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%6)
  //          %5 = h(%6, %6)
  TORCH_API void replaceAllUsesWith(Value* newValue);

  // Replaces all uses of this value with 'newValue' after 'node'.
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = inplace_(%3)
  //          %6 = h(%3, %3)
  // Execute: %3.replaceAllUsesAfterNodeWith(%5.node(), %5)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = inplace_(%3)
  //          %6 = h(%5, %5)
  // XXX: does not check scoping legality, consider using
  // replaceAllUsesDominatedByNodeWith
  TORCH_API void replaceAllUsesAfterNodeWith(const Node* node, Value* newValue);

  // Replaces all uses of this value with 'newValue' that are dominated by
  // 'node'. Given:
  // x = op(...).
  // if cond:
  //    z = foo(..)
  //    bar(x)
  // else:
  //    print(x)
  // x.replaceAllUsesDominatedByNodeWith(foo, z) would replace bar(x)
  // but not print(x) because print is not dominated by foo.
  // replaceAllUsesAfterNode does not check domination, so in this example
  // it would produce invalid IR.
  TORCH_API void replaceAllUsesDominatedByNodeWith(
      const Node* node,
      Value* newValue);

  TORCH_API Value* copyMetadata(Value* from);

  TORCH_API std::shared_ptr<Wrap<Value>> wrap() {
    if (!wrap_) {
      wrap_ = std::make_shared<Wrap<Value>>(this);
    }
    return wrap_;
  }

  virtual ~Value() {
    if (wrap_) {
      wrap_->clear();
    }
  }
};

} // namespace habana_torch::jit

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

#include <ATen/core/jit_type.h>

#include <optional>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

namespace habana_torch {
namespace jit {

using SymbolOrExpr = std::string;

using DimVariants = std::variant<std::monostate, SymbolOrExpr, int64_t>;
using SymbolicShape = std::vector<DimVariants>;
using SymbolicStrides = std::vector<DimVariants>;

class TypeWrapper {
 public:
  TypeWrapper();
  TypeWrapper(c10::TypePtr underlying_type);
  TypeWrapper(c10::TypePtr underlying_type, const SymbolOrExpr& symbol_or_expr);
  TypeWrapper(
      c10::TensorTypePtr underlying_type,
      const SymbolicShape& symbolic_shape,
      const SymbolicStrides& symbolic_strides);

  TypeWrapper& operator=(c10::TypePtr underlying_type);

  TypeWrapper(const TypeWrapper& other);
  TypeWrapper& operator=(const TypeWrapper& other);

  // Default move operations
  TypeWrapper(TypeWrapper&&) noexcept = default;
  TypeWrapper& operator=(TypeWrapper&&) noexcept = default;

  const c10::TypePtr& operator->() const;
  const c10::TypePtr& operator*() const;
  explicit operator bool() const noexcept;

  static TypeWrapper createTensorTypeWrapper(
      at::ScalarType scalar_type,
      const SymbolicShape& shape,
      const SymbolicStrides& strides,
      c10::optional<c10::Device> device,
      c10::optional<bool> requires_grad);

  const c10::TypePtr& getType() const;
  const SymbolOrExpr& getSymbolOrExpr() const;
  const SymbolicStrides& getStrides() const;
  const SymbolicShape& getShape() const;
  bool isSymbolic() const;
  bool hasSymbolicShapeOrStrides() const;

 private:
  struct TensorTypeDetails {
    SymbolicShape shape;
    SymbolicStrides strides;
  };

  using TypeDetails =
      std::variant<std::monostate, SymbolOrExpr, TensorTypeDetails>;

  c10::TypePtr initType(c10::TypePtr underlying_type);
  TypeDetails initTypeDetails(SymbolOrExpr symbol_or_expr);
  TypeDetails initTypeDetails(
      SymbolicShape symbolic_shape,
      SymbolicStrides symbolic_strides);

  c10::TypePtr type_;
  TypeDetails type_details_;
};

bool matchTypes(
    const c10::TypePtr& lhs,
    const c10::TypePtr& rhs,
    std::ostream* why_not = nullptr);

std::ostream& operator<<(std::ostream& out, const TypeWrapper& t);

// This is a wrapper to allow invalidating the Python object
// safely when the C++ object for a Node/Value/Block is deleted
// like much of graph, it isn't safe for different threads to
// access the same graph
template <typename T>
struct Wrap {
  explicit Wrap(T* p) : elem(p), clear_cb(nullptr) {}
  void clear() {
    if (clear_cb) {
      clear_cb(elem);
    }
    elem = nullptr;
  }
  T* elem;
  void (*clear_cb)(void*);
};

// Note [User node does not uniquely identify use]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// A while back, we wrote some code manipulating uses that looked like this:
//
//    for (auto& use : used_val->uses_) {
//      if (use.user == this_node) {
//        use.offset += 1;
//        break;
//      }
//    }
//
// This code is trying to find a particular use (our node's use) to update it.
// However, it's wrong: there may be *multiple* uses of a value %x in a node,
// as might be the case in this IR:
//
//    %y = Add %x %x
//
// In this case, there are two uses of %x whose user is the node 'Add %x %x'.
// So, "use induced by this node" is not a well-formed concept.
//
// If you are looking for "use induced by an input", it's best to use
// findUseForInput() to get it.

// Each use is represented by this type, see 'Node::uses()'
// 'user' is the consumer of the value, 'offset' is the index into
// 'user's input this where the producers will be found.
struct Node;
struct Use {
  Use(Node* user, size_t offset) : user(user), offset(offset) {}
  Node* user;
  size_t offset;

  bool operator==(const Use& b) {
    return user == b.user && offset == b.offset;
  }
};

} // namespace jit
} // namespace habana_torch

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

} // namespace jit
} // namespace habana_torch

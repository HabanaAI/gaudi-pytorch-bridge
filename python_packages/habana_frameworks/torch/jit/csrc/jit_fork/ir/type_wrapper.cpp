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

#include "jit_fork/ir/type_wrapper.h"

#include <ATen/core/function_schema.h>

#include <algorithm>
#include <cctype>
#include <iterator>
#include <string>

namespace habana_torch {
namespace jit {

namespace {
std::string& removeWhitespaces(std::string& str) {
  str.erase(
      std::remove_if(
          str.begin(),
          str.end(),
          [](unsigned char c) { return std::isspace(c); }),
      str.end());
  return str;
}
} // namespace

using namespace c10;

TypeWrapper::TypeWrapper() : type_(nullptr), type_details_(std::monostate{}) {}

TypeWrapper::TypeWrapper(TypePtr underlying_type)
    : type_(initType(underlying_type)), type_details_(std::monostate{}) {}

TypeWrapper::TypeWrapper(
    TypePtr underlying_type,
    const SymbolOrExpr& symbol_or_expr)
    : type_(initType(underlying_type)),
      type_details_(initTypeDetails(symbol_or_expr)) {}

TypeWrapper::TypeWrapper(
    TensorTypePtr underlying_type,
    const SymbolicShape& symbolic_shape,
    const SymbolicStrides& symbolic_strides)
    : type_(initType(underlying_type)),
      type_details_(initTypeDetails(symbolic_shape, symbolic_strides)) {}

TypeWrapper& TypeWrapper::operator=(TypePtr underlying_type) {
  type_ = initType(underlying_type);
  type_details_ = std::monostate{};

  return *this;
}

TypeWrapper::TypeWrapper(const TypeWrapper& other)
    : type_(initType(other.type_)), type_details_(other.type_details_) {}

TypeWrapper& TypeWrapper::operator=(const TypeWrapper& other) {
  if (this != &other) {
    type_ = initType(other.type_);
    type_details_ = other.type_details_;
  }

  return *this;
}

TypeWrapper TypeWrapper::createTensorTypeWrapper(
    at::ScalarType scalar_type,
    const SymbolicShape& shape,
    const SymbolicStrides& strides,
    c10::optional<Device> device,
    c10::optional<bool> requires_grad) {
  TORCH_CHECK(
      shape.size() == strides.size(),
      "The number of dimensions is not equal in shape and strides.");

  const auto has_symbolic_info = [](const DimVariants& dim) {
    return std::holds_alternative<std::string>(dim);
  };
  const bool are_strides_symbolic =
      std::any_of(strides.cbegin(), strides.cend(), has_symbolic_info);
  const bool is_shape_symbolic =
      std::any_of(shape.cbegin(), shape.cend(), has_symbolic_info);

  if (are_strides_symbolic || is_shape_symbolic) {
    TensorTypePtr tensor_type =
        TensorType::create(scalar_type, device, shape.size(), requires_grad);

    return TypeWrapper(tensor_type, shape, strides);
  } else {
    std::vector<int64_t> fixed_shape;
    std::vector<int64_t> fixed_strides;
    fixed_shape.reserve(shape.size());
    fixed_strides.reserve(strides.size());

    const auto get_fixed_value = [](const DimVariants& dim) {
      TORCH_CHECK(
          std::holds_alternative<int64_t>(dim),
          "Found non-fixed dimension for a tensor with a fixed shape.");
      return std::get<int64_t>(dim);
    };
    std::transform(
        shape.cbegin(),
        shape.cend(),
        std::back_inserter(fixed_shape),
        get_fixed_value);
    std::transform(
        strides.cbegin(),
        strides.cend(),
        std::back_inserter(fixed_strides),
        get_fixed_value);

    TensorTypePtr tensor_type = TensorType::create(
        scalar_type,
        device,
        c10::VaryingShape<int64_t>(fixed_shape),
        c10::VaryingShape<int64_t>(fixed_strides),
        requires_grad);
    return TypeWrapper(tensor_type, shape, strides);
  }
}

TypePtr TypeWrapper::initType(TypePtr underlying_type) {
  if (underlying_type) {
    if (auto dyn = underlying_type->castRaw<DynamicType>()) {
      underlying_type = dyn->fallback();
    }
  }

  return underlying_type;
}

TypeWrapper::TypeDetails TypeWrapper::initTypeDetails(
    SymbolOrExpr symbol_or_expr) {
  TORCH_CHECK(
      isSymbolic(),
      "Single symbol or expression is allowed only for simple symbolic types.");

  return removeWhitespaces(symbol_or_expr);
}

TypeWrapper::TypeDetails TypeWrapper::initTypeDetails(
    SymbolicShape symbolic_shape,
    SymbolicStrides symbolic_strides) {
  const TypeKind type_kind = type_->kind();
  const bool is_tensor_type = type_kind == TypeKind::TensorType;
  TORCH_CHECK(
      is_tensor_type,
      "Shape and strides info is allowed only for tensor type.");

  for (auto& dim : symbolic_shape) {
    if (std::holds_alternative<SymbolOrExpr>(dim)) {
      removeWhitespaces(std::get<SymbolOrExpr>(dim));
    }
  }

  for (auto& stride : symbolic_strides) {
    if (std::holds_alternative<SymbolOrExpr>(stride)) {
      removeWhitespaces(std::get<SymbolOrExpr>(stride));
    }
  }

  return TypeWrapper::TensorTypeDetails{symbolic_shape, symbolic_strides};
}

const TypePtr& TypeWrapper::operator->() const {
  return getType();
}

const TypePtr& TypeWrapper::operator*() const {
  return getType();
}

TypeWrapper::operator bool() const noexcept {
  return getType() ? true : false;
}

const TypePtr& TypeWrapper::getType() const {
  TORCH_CHECK(type_ != nullptr, "Underlying type was not initialized.");
  return type_;
}

const SymbolOrExpr& TypeWrapper::getSymbolOrExpr() const {
  TORCH_CHECK(
      isSymbolic(),
      "Single symbol or expression is allowed only for simple symbolic types.");
  return std::get<SymbolOrExpr>(type_details_);
}

const SymbolicStrides& TypeWrapper::getStrides() const {
  const TypeKind type_kind = type_->kind();
  const bool is_tensor_type = type_kind == TypeKind::TensorType;
  TORCH_CHECK(is_tensor_type, "Strides info is allowed only for tensor type.");
  return std::get<TensorTypeDetails>(type_details_).strides;
}

const SymbolicShape& TypeWrapper::getShape() const {
  const TypeKind type_kind = type_->kind();
  const bool is_tensor_type = type_kind == TypeKind::TensorType;
  TORCH_CHECK(is_tensor_type, "Shape info is allowed only for tensor type.");
  return std::get<TensorTypeDetails>(type_details_).shape;
}

bool TypeWrapper::isSymbolic() const {
  const TypeKind type_kind = type_->kind();
  return type_kind == TypeKind::SymIntType ||
      type_kind == TypeKind::SymFloatType || type_kind == TypeKind::SymBoolType;
}

bool TypeWrapper::hasSymbolicShapeOrStrides() const {
  if (type_->kind() == TypeKind::TensorType) {
    const auto& shape = getShape();
    const auto& strides = getStrides();

    const auto has_symbolic_info = [](const DimVariants& dim) {
      return std::holds_alternative<std::string>(dim);
    };
    const bool are_strides_symbolic =
        std::any_of(strides.cbegin(), strides.cend(), has_symbolic_info);
    const bool is_shape_symbolic =
        std::any_of(shape.cbegin(), shape.cend(), has_symbolic_info);

    return are_strides_symbolic || is_shape_symbolic;
  }

  return false;
}

bool matchTypes(const TypePtr& lhs, const TypePtr& rhs, std::ostream* why_not) {
  auto get_fake_type = [](const TypePtr& type) -> TypePtr {
    if (type->kind() == TypeKind::SymIntType) {
      return IntType::get();
    } else if (type->kind() == TypeKind::SymFloatType) {
      return FloatType::get();
    } else if (type->kind() == TypeKind::SymBoolType) {
      return BoolType::get();
    }

    return type;
  };

  auto is_match = [](const TypePtr& lhs,
                     const TypePtr& rhs,
                     std::ostream* why_not = nullptr) {
    return lhs->isSubtypeOfExt(*rhs, why_not);
  };

  if (!is_match(get_fake_type(lhs), get_fake_type(rhs), why_not)) {
    ListTypePtr rhs_list = lhs->cast<ListType>();
    ListTypePtr lhs_list = rhs->cast<ListType>();
    if (rhs_list && lhs_list) {
      return is_match(
          get_fake_type(rhs_list->getElementType()),
          get_fake_type(lhs_list->getElementType()));
    }

    OptionalTypePtr rhs_optional = lhs->cast<OptionalType>();
    OptionalTypePtr lhs_optional = rhs->cast<OptionalType>();
    if (rhs_optional && lhs_optional) {
      return is_match(
          get_fake_type(rhs_list->getElementType()),
          get_fake_type(lhs_list->getElementType()));
    }

    TupleTypePtr rhs_tuple = lhs->cast<TupleType>();
    TupleTypePtr lhs_tuple = rhs->cast<TupleType>();
    if (rhs_tuple && lhs_tuple) {
      const auto& lhs_types = lhs_tuple->containedTypes();
      const auto& rhs_types = rhs_tuple->containedTypes();
      if (lhs_types.size() == rhs_types.size()) {
        for (unsigned i = 0; i < lhs_types.size(); i++) {
          if (!is_match(
                  get_fake_type(lhs_types[i]), get_fake_type(rhs_types[i]))) {
            return false;
          }
        }
        return true;
      }
    }

    return false;
  }

  return true;
}

std::ostream& operator<<(std::ostream& out, const TypeWrapper& wrapper) {
  if (auto value = wrapper->cast<TensorType>()) {
    if (value->scalarType().has_value()) {
      out << toString(*value->scalarType());
      if (!value->sizes().size().has_value()) {
        out << "Tensor";
      }
    } else {
      out << "Tensor";
    }
    if (auto ndim = value->sizes().size()) {
      const auto& sym_shape = wrapper.getShape();
      const auto& sym_strides = wrapper.getStrides();

      out << "(shape=[";
      for (size_t i = 0; i < *ndim; ++i) {
        if (i > 0) {
          out << ", ";
        }
        if (std::holds_alternative<std::string>(sym_shape[i])) {
          out << std::get<std::string>(sym_shape[i]);
        } else if ((std::holds_alternative<int64_t>(sym_shape[i]))) {
          out << std::get<int64_t>(sym_shape[i]);
        } else {
          TORCH_CHECK(0, "Shape contains incomplete dimension info.");
        }
      }
      out << "]";
      out << ", strides=[";
      for (size_t i = 0; i < *ndim; ++i) {
        if (i > 0) {
          out << ", ";
        }
        if (std::holds_alternative<std::string>(sym_strides[i])) {
          out << std::get<std::string>(sym_strides[i]);
        } else if (std::holds_alternative<int64_t>(sym_strides[i])) {
          out << std::get<int64_t>(sym_strides[i]);
        } else {
          TORCH_CHECK(0, "Strides contain incomplete dimension info.");
        }
      }
      out << "]";
      if (value->requiresGrad()) {
        out << ", ";
        out << "requires_grad=" << *value->requiresGrad();
      }
      if (value->device()) {
        out << ", ";
        out << "device=" << *value->device();
      }
      out << ")";
    } else {
      size_t i = 0;
      if (value->requiresGrad()) {
        out << "("
            << "requires_grad=" << *value->requiresGrad();
        i++;
      }
      if (value->device()) {
        out << ((i++ > 0) ? ", " : "(") << "device=" << *value->device();
      }
      if (i > 0) {
        out << ")";
      }
    }

    if (value->undefined() && *value->undefined()) {
      out << "[Undefined]";
    }
  } else if (wrapper->kind() == TypeKind::ListType) {
    auto prim = wrapper->castRaw<ListType>()->getElementType();
    out << *prim << "[]";
  } else if (wrapper->kind() == TypeKind::OptionalType) {
    auto prim = wrapper->castRaw<OptionalType>()->getElementType();
    out << *prim << "?";
  } else if (wrapper->kind() == TypeKind::FutureType) {
    auto elem = wrapper->castRaw<FutureType>()->getElementType();
    out << "Future[" << *elem << "]";
  } else if (wrapper->kind() == TypeKind::RRefType) {
    auto elem = wrapper->castRaw<RRefType>()->getElementType();
    out << "RRef[" << *elem << "]";
  } else if (auto tup = wrapper->cast<TupleType>()) {
    if (tup->schema()) {
      out << "NamedTuple";
    }
    out << "(";
    for (size_t i = 0; i < tup->elements().size(); ++i) {
      if (i > 0)
        out << ", ";
      if (tup->schema()) {
        auto arg = tup->schema()->arguments()[i];
        out << arg.name() << " : ";
        out << *(tup->elements()[i]);
        if (arg.default_value()) {
          out << " = " << *arg.default_value();
        }
      } else {
        out << *(tup->elements()[i]);
      }
    }
    out << ")";
  } else if (wrapper->kind() == TypeKind::FunctionType) {
    out << "Function";
  } else if (
      wrapper->kind() == TypeKind::SymIntType ||
      wrapper->kind() == TypeKind::SymFloatType ||
      wrapper->kind() == TypeKind::SymBoolType) {
    out << wrapper->str() << "(" << wrapper.getSymbolOrExpr() << ")";
  } else {
    out << wrapper->str();
  }
  return out;
}

} // namespace jit
} // namespace habana_torch

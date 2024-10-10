/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#pragma once

#include <torch/torch.h>
#include <iostream>
#include <variant>
#include <vector>
#include "backend/helpers/symbolic_expression.h"

namespace habana_helpers {

struct RangeInfo {
  std::vector<int64_t> min_shape;
  std::vector<int64_t> max_shape;
  std::string expr;
  std::string expr_strides;
  int index;

  RangeInfo(
      std::vector<int64_t> min_shape,
      std::vector<int64_t> max_shape,
      std::string expr,
      std::string expr_strides,
      int index)
      : min_shape(min_shape),
        max_shape(max_shape),
        expr(expr),
        expr_strides(expr_strides),
        index(index) {}
};

// Enum to represent different data types stored in IValue
enum class IShapeType {
  SCALAR,
  TENSOR_SHAPE,
  NONE,
};

// Class to represent IValue
class IShape {
 private:
  IShapeType type;
  std::variant<at::Scalar, std::vector<int64_t>> data;
  c10::ScalarType scalarType;
  bool updated;

 public:
  // Constructors

  IShape() : type(IShapeType::NONE) {}

  IShape(at::Scalar scalar, c10::ScalarType scalarType = c10::ScalarType::Int)
      : type(IShapeType::SCALAR),
        data(scalar),
        scalarType(scalarType),
        updated(false) {}

  IShape(std::vector<int64_t> tensorShape, c10::ScalarType scalarType)
      : type(IShapeType::TENSOR_SHAPE),
        data(tensorShape),
        scalarType(scalarType),
        updated(false) {}

  // Getters
  IShapeType getType() const;
  at::Scalar getScalar() const;
  std::vector<int64_t> getTensorShape() const;
  c10::ScalarType toScalarType() const;
  c10::ScalarType getScalarType() const;
  void UpdateTensor(std::vector<int64_t> tensorShape);
  void UpdateScalar(at::Scalar scalar);
  void ResetIshapeUpdate();
  bool IsUpdated();
  bool isScalar();
  bool isTensor();
};

typedef std::vector<IShape> IShapeList;

using DSInputSymbolMap =
    std::unordered_map<std::string, std::shared_ptr<double>>;
using DSValueExprMap = std::unordered_map<
    std::string,
    std::vector<std::shared_ptr<habana::SizeExpression>>>;
using DSValueIShapeMap =
    std::unordered_map<std::string, habana_helpers::IShape>;

struct DynamicSIFInfo {
  DSInputSymbolMap expr_symbolic_table;
  DSValueExprMap value_to_sizeexpr{};
  DSValueIShapeMap value_to_ishape;
};

void UpdateSTShapeInfo(std::vector<int64_t>& shape);
bool is_symbolic_expr(std::string expr_str);

} // namespace habana_helpers

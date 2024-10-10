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

#include "backend/helpers/dynamic_shape_infer.h"
#include "backend/kernel/hpu_shape_inference.h"

namespace habana_helpers {

// Getters
IShapeType IShape::getType() const {
  return type;
}

at::Scalar IShape::getScalar() const {
  if (type == IShapeType::SCALAR) {
    return std::get<at::Scalar>(data);
  } else {
    throw std::runtime_error("Get scalar failed!!!");
  }
}

std::vector<int64_t> IShape::getTensorShape() const {
  if (type == IShapeType::TENSOR_SHAPE) {
    return std::get<std::vector<int64_t>>(data);
  } else {
    throw std::runtime_error("Get tensor shape failed!!!");
  }
}

c10::ScalarType IShape::getScalarType() const {
  return scalarType;
}

bool IShape::isScalar() {
  return type == IShapeType::SCALAR;
}

bool IShape::isTensor() {
  return type == IShapeType::TENSOR_SHAPE;
}

bool IShape::IsUpdated() {
  return updated;
}

void IShape::ResetIshapeUpdate() {
  updated = false;
}

void IShape::UpdateTensor(std::vector<int64_t> tensorShape) {
  if (type == IShapeType::TENSOR_SHAPE) {
    data = tensorShape;
    updated = true;
  } else {
    throw std::runtime_error("Update tensor shape failed!!!");
  }
}

void IShape::UpdateScalar(at::Scalar scalar) {
  if (type == IShapeType::SCALAR) {
    data = scalar;
    updated = true;
  } else {
    throw std::runtime_error("Update scalar failed!!!");
  }
}

c10::ScalarType IShape::toScalarType() const {
  if (type == IShapeType::SCALAR) {
    auto scalar_data = std::get<at::Scalar>(data);
    return torch::jit::IValue(scalar_data).toScalarType();
  } else {
    throw std::runtime_error("To Scalar failed!!!");
  }
}

void UpdateSTShapeInfo(std::vector<int64_t>& shape) {
  uint64_t shape_tensor_id =
      habana::ShapeInference::ReadAndIncrementShapeTensorId();
  uint64_t tensor_id =
      habana::ShapeInference::GetSTMappedTensorIdx(shape_tensor_id);
  PT_DYNAMIC_SHAPE_DEBUG(
      "OUTPUT PASS: ST_ID = ", shape_tensor_id, " TID = ", tensor_id);
  habana::ShapeInference::UpdateShapeInfoDynamic(tensor_id, shape);
}

bool is_symbolic_expr(std::string expr_str) {
  for (auto& c : expr_str) {
    if (!(std::isdigit(c) || c == '[' || c == ']' || c == ',' ||
          std::isspace(c)))
      return true;
  }
  return false;
}

} // namespace habana_helpers

/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "scalar_cache.h"
#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include "backend/helpers/tensor_utils.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace habana {

namespace backend {

c10::ScalarType GetInternalScalarType(const c10::ScalarType& scalar_type) {
  switch (scalar_type) {
    case c10::ScalarType::Long:
      return c10::ScalarType::Int;
    case c10::ScalarType::Double:
      return c10::ScalarType::Float;
    case c10::ScalarType::Bool:
      return scalar_type;
    default:
      break;
  }

  HABANA_ASSERT(0, "Not supported scalar type");
}

at::Tensor ScalarCache::AppendToBatchH2DList(const at::Tensor& scalar_tensor) {
  auto scalar_type = scalar_tensor.scalar_type();
  auto internal_scalar_type = GetInternalScalarType(scalar_type);

  auto t = at::empty(
      {},
      scalar_tensor.options()
          .dtype(internal_scalar_type)
          .device(c10::DeviceType::HPU));
  t.unsafeGetTensorImpl()->set_wrapped_number(true);
  auto tensor = scalar_tensor.to(internal_scalar_type);
  copy_tensor_list_.emplace_back(tensor, t);
  return t;
}

template <typename T>
at::Tensor ScalarCache::GetTensorFromCacheMap(
    std::unordered_map<T, at::Tensor>& map,
    const T value,
    const c10::ScalarType& dtype) {
  auto it = map.find(value);
  if (it == map.end()) {
    auto tensor = AppendToBatchH2DList(at::tensor(value).to(dtype));
    auto ret = map.insert(std::make_pair(value, tensor));
    return ret.first->second;
  }
  return it->second;
}

at::Tensor ScalarCache::GetTensor(const at::Scalar& scalar) {
  auto dtype = scalar.type();

  switch (dtype) {
    case c10::ScalarType::Double:
      return GetTensorFromCacheMap(double_to_tensor_, scalar.toDouble(), dtype);
    case c10::ScalarType::Long:
      return GetTensorFromCacheMap(int64_to_tensor_, scalar.toLong(), dtype);
    case c10::ScalarType::Bool:
      return GetTensorFromCacheMap(
          int8_to_tensor_,
          static_cast<int8_t>(scalar.toBool()),
          dtype); // at:tensor(value), doesn't overload for bool value
    default:
      HABANA_ASSERT(0, "Not supported scalar type");
  }
}

void ScalarCache::CopyScalarsToDevice() {
  if (copy_tensor_list_.empty()) {
    return;
  }

  habana_helpers::copy_scalars_to_device(copy_tensor_list_);
  copy_tensor_list_.clear();
}

void ScalarCache::ClearCache() {
  copy_tensor_list_.clear();
  int64_to_tensor_.clear();
  double_to_tensor_.clear();
  int8_to_tensor_.clear();
}

} // namespace backend
} // namespace habana
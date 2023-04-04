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

at::Tensor ScalarCache::GetTensor(const at::Scalar& scalar) {
  auto dtype = scalar.type();
  if (dtype == c10::ScalarType::Double) {
    auto value = scalar.toDouble();
    auto it = double_to_tensor_.find(value);
    if (it == double_to_tensor_.end()) {
      auto tensor = AppendToBatchH2DList(at::tensor(value).to(dtype));
      auto ret = double_to_tensor_.insert(std::make_pair(value, tensor));
      return ret.first->second;
    }

    return it->second;
  }

  if (dtype == c10::ScalarType::Long) {
    auto value = scalar.toLong();
    auto it = int64_to_tensor_.find(value);
    if (it == int64_to_tensor_.end()) {
      auto tensor = AppendToBatchH2DList(at::tensor(value).to(dtype));
      auto ret = int64_to_tensor_.insert(std::make_pair(value, tensor));
      return ret.first->second;
    }

    return it->second;
  }

  HABANA_ASSERT(0, "Not supported scalar type");
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
}

} // namespace backend
} // namespace habana
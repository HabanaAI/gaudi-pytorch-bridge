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

#pragma once

#include <ATen/core/Tensor.h>
#include <deque>
#include <unordered_map>

namespace habana {

namespace backend {

class ScalarCache {
 public:
  ScalarCache() = default;
  ScalarCache(const ScalarCache&) = delete;
  ScalarCache(ScalarCache&&) = delete;
  ScalarCache& operator=(const ScalarCache&) = delete;
  ScalarCache& operator=(ScalarCache&&) = delete;
  ~ScalarCache() = default;

  at::Tensor GetTensor(const at::Scalar& scalar);
  void CopyScalarsToDevice();
  void ClearCache();

 private:
  template <typename T>
  at::Tensor GetTensorFromCacheMap(
      std::unordered_map<T, at::Tensor>& map,
      const T value,
      const c10::ScalarType& dtype);

  std::vector<std::pair<at::Tensor, at::Tensor>> copy_tensor_list_;
  std::unordered_map<int64_t, at::Tensor> int64_to_tensor_;
  std::unordered_map<double, at::Tensor> double_to_tensor_;
  std::unordered_map<int8_t, at::Tensor> int8_to_tensor_;

  at::Tensor AppendToBatchH2DList(const at::Tensor& scalar_tensor);
};

} // namespace backend
} // namespace habana
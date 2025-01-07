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
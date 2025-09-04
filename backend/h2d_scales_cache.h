/**
 * Copyright (c) 2025 Intel Corporation
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
#include <unordered_map>

namespace habana::backend {

struct HashFn {
  std::size_t operator()(const std::pair<double, at::ScalarType>& pair) const {
    return std::hash<double>()(pair.first) ^
        std::hash<float>()((float)pair.second);
  }
};

class EqualFn {
 public:
  bool operator()(
      const std::pair<double, at::ScalarType>& a,
      const std::pair<double, at::ScalarType>& b) const {
    return a.first == b.first && a.second == b.second;
  }
};

using ScalarToScalesMap = std::unordered_map<
    std::pair<double, at::ScalarType>,
    at::Tensor,
    HashFn,
    EqualFn>;

class H2dScalesCache {
 public:
  H2dScalesCache() = default;
  H2dScalesCache(const H2dScalesCache&) = delete;
  H2dScalesCache(H2dScalesCache&&) = delete;
  H2dScalesCache& operator=(const H2dScalesCache&) = delete;
  H2dScalesCache& operator=(H2dScalesCache&&) = delete;
  ~H2dScalesCache() = default;

  bool CreateH2dScales();
  std::optional<at::Tensor> TryGetH2dScale(
      const double scale,
      const at::ScalarType dtype) const;
  static at::Tensor CreateH2dTensorScale(
      void* scale_ptr,
      at::ScalarType dtype,
      void*& alloc_pointer,
      void*& h2d_pointer);

 private:
  ScalarToScalesMap h2d_scales_map_;
};

} // namespace habana::backend

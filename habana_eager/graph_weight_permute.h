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

#include <ATen/Tensor.h>

#include "backend/backend_meta.h"

namespace habana {
namespace graph {
struct PermuteWeightTensor {
  explicit PermuteWeightTensor(const torch::Tensor& weight);
  void PermuteIfNeeded();

 private:
  bool ShouldPermuteWeight();
  template <typename T>
  void PermuteDataToRSCK(const torch::Tensor& weight_cpu);
  template <typename T>
  void PermuteDataToQRSCK(const torch::Tensor& weight_cpu);
  const torch::Tensor& m_weight;
  int64_t m_tensor_dim;
  StorageExtraMeta* m_storage_meta;
};

} // namespace graph
} // namespace habana
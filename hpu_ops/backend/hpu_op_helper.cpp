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
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

std::vector<at::Tensor> GetMetaTensorList(
    const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> metatensors;
  metatensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    metatensors.emplace_back(at::empty(
        tensor.sizes(),
        tensor.options().device(at::kMeta),
        tensor.suggest_memory_format()));
  }
  return metatensors;
}

std::vector<c10::optional<at::Tensor>> GetMetaOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors) {
  std::vector<c10::optional<at::Tensor>> metatensors;
  metatensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    if (tensor.has_value()) {
      const auto& tv = tensor.value();
      metatensors.emplace_back(at::empty(
          tv.sizes(),
          tv.options().device(at::kMeta),
          tv.suggest_memory_format()));
    } else {
      metatensors.emplace_back(tensor);
    }
  }
  return metatensors;
}

} // namespace habana

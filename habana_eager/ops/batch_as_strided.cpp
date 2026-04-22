/**
 * Copyright (c) 2024-2025 Intel Corporation
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

#include "habana_eager/ops/batch_as_strided.h"
#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Tensor.h>
#include <torch/library.h>
#include <vector>
#include "habana_eager/ops/as_strided.h"
#include "habana_helpers/logging.h"

namespace habana::eager {
std::vector<at::Tensor> batch_as_strided(
    at::TensorList inputs,
    c10::ArrayRef<std::vector<int64_t>> sizes,
    c10::ArrayRef<std::vector<int64_t>> strides,
    at::OptionalIntArrayRef storage_offsets) {
  auto inputs_count = inputs.size();
  HABANA_ASSERT(
      sizes.size() == inputs_count,
      "Length of sizes array doesn't match the number of provided input tensors");
  HABANA_ASSERT(
      strides.size() == inputs_count,
      "Length of strides array doesn't match the number of provided input tensors");
  if (storage_offsets.has_value()) {
    HABANA_ASSERT(
        storage_offsets.value().size() == inputs_count,
        "Length of storage_offsets array doesn't match the number of provided input tensors");
  }
  std::vector<at::Tensor> outputs;
  outputs.reserve(inputs_count);
  for (size_t i = 0; i < inputs_count; i++) {
    outputs.push_back(
        at::as_strided(
            inputs[i],
            sizes[i],
            strides[i],
            storage_offsets.has_value() ? storage_offsets.value()[i] : 0));
  }
  return outputs;
}

TORCH_LIBRARY_IMPL(hpu, AutogradHPU, m) {
  m.impl("hpu::batch_as_strided", batch_as_strided);
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("hpu::batch_as_strided", batch_as_strided);
}

} // namespace habana::eager

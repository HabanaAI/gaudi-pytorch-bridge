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

#include <cstdint>
#include <vector>

#include <torch/torch.h>

namespace slrg {
at::Tensor createTensor(
    std::int64_t rank,
    at::ScalarType dtype,
    const int dim_size = 1) {
  std::vector<std::int64_t> shape(rank, dim_size);
  return at::zeros(shape, torch::TensorOptions().dtype(dtype));
}
} // namespace slrg

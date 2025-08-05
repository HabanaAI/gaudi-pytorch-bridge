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

#include <vector>
#include "habana_eager/ops/as_strided.h"
#include "habana_eager/ops/batch_as_strided.h"

namespace habana::eager {
std::vector<at::Tensor> batch_as_strided(
    at::TensorList inputs,
    c10::ArrayRef<std::vector<int64_t>> sizes,
    c10::ArrayRef<std::vector<int64_t>> strides,
    at::OptionalIntArrayRef storage_offsets);
} // namespace habana::eager

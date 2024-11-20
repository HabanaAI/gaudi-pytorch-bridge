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

#include "hpu_ops/hpu_op_helper.h"

namespace habana {

constexpr size_t index_of_self = 0;
constexpr size_t index_of_other = 1;
static inline void ScalarTypeConvert(
    std::vector<at::IValue>& inputs,
    size_t scalar_index,
    size_t tensor_index) {
  auto tensor = inputs[tensor_index].toTensor();
  auto scalar = inputs[scalar_index].toScalar();
  if (c10::isFloatingType(tensor.scalar_type())) {
    inputs[scalar_index] = scalar.to<float>();
  } else {
    inputs[scalar_index] = scalar.to<int>();
  }
}
} // namespace habana

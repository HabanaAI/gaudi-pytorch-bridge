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

#include <torch/torch.h>

namespace habana {

template <typename T>
bool is_out_of_dtype_range(const float value);

bool is_value_out_of_scalar_range(
    const float value,
    const c10::ScalarType scalar_type);

void update_other_scalar_if_out_of_scalar_type_range(
    const std::vector<at::IValue>& inputs,
    std::vector<at::IValue>& hpu_inputs);

} // namespace habana
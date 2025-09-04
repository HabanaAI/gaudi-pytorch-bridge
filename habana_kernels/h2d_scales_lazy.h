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
#include <ATen/Tensor.h>

namespace habana_lazy {

std::optional<at::Tensor> maybe_convert_tensor_to_h2d(
    const std::optional<at::Tensor>& tensor,
    const std::string_view op_name);

std::optional<at::Tensor> maybe_convert_tensor_to_h2d(
    const std::optional<at::Tensor>& tensor,
    const bool enabled,
    const std::string_view op_name);

std::vector<at::Tensor> maybe_convert_list_to_h2d(
    const at::TensorList& tensors,
    const std::string_view op_name);

void verify_no_h2d_scales(
    const std::vector<at::TensorList>& scales_lists,
    std::string_view op_name);

} // namespace habana_lazy

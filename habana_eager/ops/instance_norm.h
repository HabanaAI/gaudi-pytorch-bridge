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

#include <ATen/core/TensorBody.h>

namespace habana {
namespace eager {

at::Tensor instance_norm_autograd_wrap(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    bool use_input_stats,
    double momentum,
    double eps,
    bool cudnn_enabled);

std::tuple<at::Tensor, at::Tensor, at::Tensor> instance_norm_fwd_eager_hpu(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    double eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> instance_norm_bwd_eager_hpu(
    const at::Tensor& input,
    const at::Tensor& grad_in,
    const at::Tensor& mean,
    const at::Tensor& istd,
    const c10::optional<at::Tensor>& gamma);

} // namespace eager
} // namespace habana

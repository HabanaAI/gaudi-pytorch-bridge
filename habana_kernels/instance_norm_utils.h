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
/*******************************************************************************
 * Core content of this file comes from:
 * pytorch-fork/torch/csrc/autograd/FunctionsManual.h
 * It was modified by adding at:: namespace to all Tensor classes
 * License available: python_packages/LICENSE.txt
 *******************************************************************************
 */
#pragma once

std::tuple<at::Tensor, at::Tensor, at::Tensor> batchnorm_double_backward(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& gamma,
    const at::Tensor& ggI,
    const at::Tensor& ggG,
    const at::Tensor& ggB,
    const at::Tensor& gO,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    bool training,
    double eps,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_invstd,
    std::array<bool, 3> output_mask);
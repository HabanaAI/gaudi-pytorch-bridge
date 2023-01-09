/*******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/util/Optional.h>
#include <tuple>

namespace habana_lazy {

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_hpu_lazy(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_group_norm_backward_hpu_lazy(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    std::array<bool, 3> output_mask);

} // namespace habana_lazy

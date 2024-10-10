/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

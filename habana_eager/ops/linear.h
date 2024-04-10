/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

at::Tensor linear_autograd_wrap(
    const at::Tensor& input,
    const at::Tensor& other,
    const c10::optional<at::Tensor>& bias);

at::Tensor linear_fwd_eager_hpu(
    const at::Tensor& input,
    const at::Tensor& other,
    const c10::optional<at::Tensor>& bias);

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_bwd_eager_hpu(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& other,
    std::array<bool, 3> output_mask);

} // namespace eager
} // namespace habana

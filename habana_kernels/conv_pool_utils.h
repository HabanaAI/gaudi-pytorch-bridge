/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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

#include <ATen/InferSize.h>
#include <torch/script.h>
#include <vector>

#define TRANSPOSE_IMPLEMENTED false

namespace habana_helpers {
int64_t compute_output_size(
    const int64_t input,
    const int64_t pad,
    const int64_t dilation,
    const int64_t filter,
    const int64_t stride,
    const int64_t output_pad,
    const bool ceil_mode,
    const bool transposed);

void check_convolution_params(
    const std::vector<at::Tensor>& inputs,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    const bool transposed,
    const int64_t groups,
    const int input_channel = 1,
    const int weight_channel = 1,
    const bool is_conv_3d = false);
} // namespace habana_helpers
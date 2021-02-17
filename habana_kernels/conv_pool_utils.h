/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
    const int64_t filter,
    const int64_t stride,
    const bool ceil_mode,
    const bool transposed);

void check_pool_params(
    const at::Tensor& input,
    const at::IntArrayRef kernel,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    bool ceil_mode);

void check_convolution_params(
    const std::vector<at::Tensor>& inputs,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    const bool transposed,
    const at::IntArrayRef output_padding,
    const int64_t groups,
    const int input_channel = 1,
    const int weight_channel = 1);

std::vector<int64_t> hack_pytorch_nhwc_shapes(
    const at::IntArrayRef& sizes,
    bool hack_shapes);
} // namespace habana_helpers
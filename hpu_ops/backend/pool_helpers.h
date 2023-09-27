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

#include <ATen/Tensor.h>
#include <cstdint>
#include <vector>

namespace habana {
/**
 * @brief Compute shape for output tensor(s) from given input tensor shape
 *         & pooling params such as kernel, stride, pad, dilation, ceil_mode
 */
std::vector<int64_t> compute_pool_kernel_output_shape(
    const at::Tensor& input,
    const at::IntArrayRef kernel_size,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    bool ceil_mode,
    bool is_3d = false);
} // namespace habana
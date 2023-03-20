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

#include <ATen/native/Pool.h>

#include "backend/synapse_helpers/layout_utils.h"
#include "hpu_ops/backend/pool_helpers.h"

namespace habana {
std::vector<int64_t> compute_pool_kernel_output_shape(
    const at::Tensor& input,
    const at::IntArrayRef kernel_size,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    bool ceil_mode) {
  const int filter_H = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
  const int filter_W = kernel_size.size() == 1
      ? filter_H
      : at::native::safe_downcast<int, int64_t>(kernel_size[1]);

  const int stride_H = stride.empty()
      ? filter_H
      : at::native::safe_downcast<int, int64_t>(stride[0]);
  const int stride_W = stride.empty()
      ? filter_W
      : stride.size() == 1 ? stride_H
                           : at::native::safe_downcast<int, int64_t>(stride[1]);

  const int pad_H = at::native::safe_downcast<int, int64_t>(padding[0]);
  const int pad_W = padding.size() == 1
      ? pad_H
      : at::native::safe_downcast<int, int64_t>(padding[1]);

  const int dilation_H = at::native::safe_downcast<int, int64_t>(dilation[0]);
  const int dilation_W = dilation.size() == 1
      ? dilation_H
      : at::native::safe_downcast<int, int64_t>(dilation[1]);

  // input NCHW, output NHWC
  // weight KCHW, where K - output channels
  // pad, stride HW
  unsigned int input_dim0 = 0;
  unsigned int input_dim1 = 1;
  unsigned int input_dim2 = 2;
  unsigned int input_dim3 = 3;

  input_dim0 = synapse_helpers::layouts::INPUT_N_IDX;
  input_dim1 = synapse_helpers::layouts::INPUT_C_IDX;
  input_dim2 = synapse_helpers::layouts::INPUT_H_IDX;
  input_dim3 = synapse_helpers::layouts::INPUT_W_IDX;

  const int64_t N = input.size(input_dim0);
  const int64_t C = input.size(input_dim1);
  const int64_t input_H = input.size(input_dim2);
  const int64_t input_W = input.size(input_dim3);

  const int64_t output_H = at::native::pooling_output_shape<int64_t>(
      input_H, filter_H, pad_H, stride_H, dilation_H, ceil_mode);
  const int64_t output_W = at::native::pooling_output_shape<int64_t>(
      input_W, filter_W, pad_W, stride_W, dilation_W, ceil_mode);

  return {N, C, output_H, output_W};
}
} // namespace habana
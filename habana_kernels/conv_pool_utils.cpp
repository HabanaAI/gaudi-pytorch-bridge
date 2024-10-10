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
#include <algorithm>

#include "conv_pool_utils.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/logging_pt.h"

namespace habana_helpers {

int64_t compute_output_size(
    const int64_t input,
    const int64_t pad,
    const int64_t dilation,
    const int64_t filter,
    const int64_t stride,
    const int64_t output_pad,
    const bool ceil_mode,
    const bool transposed) {
  TORCH_CHECK(!ceil_mode, "ceil_mode is not yet supported");
  if (!transposed) {
    return (input + 2 * pad - dilation * (filter - 1) - 1) / stride + 1;
  } else {
    // conv2d fwd output shape computation done as per formula provided below
    // https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
    return (
        (input - 1) * stride - 2 * pad + dilation * (filter - 1) + output_pad +
        1);
  }
}

void check_convolution_params(
    const std::vector<at::Tensor>& inputs,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    const bool transposed,
    const int64_t groups,
    const int input_channel,
    const int weight_channel,
    const bool is_conv_3d) {
  at::Tensor input = inputs[0];
  at::Tensor weight = inputs[1];
  TORCH_CHECK(
      weight.device().type() == c10::DeviceType::HPU,
      "weight is not habana at::Tensor");
  int64_t weight_dims = is_conv_3d ? 5 : 4;
  TORCH_CHECK(
      weight.ndimension() == weight_dims,
      "weight tensordimension count  != ",
      weight_dims);

  if (!transposed) {
    TORCH_CHECK(
        groups * weight.size(weight_channel) == input.size(input_channel),
        "Number of input channels doesn't match weight channels times groups ",
        "weight_channel = ",
        weight_channel,
        "input_channel = ",
        input_channel,
        weight.sizes().vec(),
        " ",
        input.sizes().vec(),
        " groups = ",
        groups);
  }

  if (inputs.size() > 2) {
    at::Tensor bias = inputs[2];
    TORCH_CHECK(
        bias.device().type() == c10::DeviceType::HPU,
        "bias is not habana at::Tensor");
    TORCH_CHECK(bias.dim() == 1, "bias at::Tensor idimension count  != 1");
  }

  TORCH_CHECK(
      std::all_of(
          dilation.cbegin(), dilation.cend(), [](int64_t x) { return x >= 1; }),
      "convolution_hpu doesn't support dilation with given dilation factor");
  TORCH_CHECK(
      input.device().type() == c10::DeviceType::HPU,
      "input is not habana at::Tensor");
  size_t stride_size = is_conv_3d ? 3 : 2;
  size_t padding_size = stride_size;
  int64_t input_dims = weight_dims;
  TORCH_CHECK(
      stride.size() == stride_size,
      "stride size != ",
      stride_size,
      " unsupported by convolution_hpu");
  TORCH_CHECK(
      padding.size() == padding_size,
      "padding size != ",
      padding_size,
      " unsupported by convolution_hpu");
  TORCH_CHECK(
      input.ndimension() == input_dims,
      "input at::Tensor dimension count !=  ",
      input_dims);
}

} // namespace habana_helpers

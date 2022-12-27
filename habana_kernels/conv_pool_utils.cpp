/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <algorithm>

#include "conv_pool_utils.h"
#include "habana_device/HPUCheck.h"
#include "habana_helpers/logging.h"

namespace habana_helpers {

int64_t compute_output_size(
    const int64_t input,
    const int64_t pad,
    const int64_t dilation,
    const int64_t filter,
    const int64_t stride,
    const bool ceil_mode,
    const bool transposed) {
  TORCH_CHECK(!ceil_mode, "ceil_mode is not yet supported");
  if (!transposed) {
    return (input + 2 * pad - dilation * (filter - 1) - 1) / stride + 1;
  } else {
    // conv2d fwd output shape computation done as per formula provided below
    // https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
    return ((input - 1) * stride - 2 * pad + dilation * (filter - 1) + 1);
  }
}

void check_pool_params(
    const at::Tensor& input,
    const at::IntArrayRef kernel,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(
      input.device().type() == c10::DeviceType::HPU,
      "input is not habana at::Tensor");

  TORCH_CHECK(
      (input.ndimension() == 4),
      "pool2d: non-empty 4D tensor expected for input");

  static_cast<void>(ceil_mode);

  TORCH_CHECK(
      kernel.size() == 1 || kernel.size() == 2,
      "pool2d: kernel_size must either be a single int, or a tuple of two ints");

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "pool2d: stride must either be omitted, a single int, or a tuple of two ints");

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "pool2d: padding must either be a single int, or a tuple of two ints");

  TORCH_CHECK(
      std::all_of(
          dilation.cbegin(), dilation.cend(), [](int64_t x) { return x == 1; }),
      "pool2d: dilation not supported, only valid value is 1");
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

std::vector<int64_t> hack_pytorch_nhwc_shapes(
    const at::IntArrayRef& sizes,
    bool hack_shapes) {
  if (hack_shapes)
    // pytorch data format is NCHW, synapse require NHWC
    return std::vector<int64_t>{sizes[0], sizes[2], sizes[3], sizes[1]};
  else
    return sizes.vec();
};

} // namespace habana_helpers

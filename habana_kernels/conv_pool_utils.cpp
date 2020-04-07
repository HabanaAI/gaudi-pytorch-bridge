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

namespace habana_helpers {

int64_t compute_output_size(
    const int64_t input,
    const int64_t pad,
    const int64_t filter,
    const int64_t stride,
    const bool ceil_mode) {
  TORCH_CHECK(!ceil_mode, "ceil_mode is not yet supported");
  return (input + 2 * pad - filter) / stride + 1;
}

void check_pool_params(
    const at::Tensor& input,
    const at::IntArrayRef kernel,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    bool ceil_mode) {

  TORCH_CHECK(
      input.device().type() == c10::DeviceType::HABANA,
      "input is not habana at::Tensor");

  TORCH_CHECK(
      (input.ndimension() == 4),
      "pool2d: non-empty 4D tensor expected for input");

  TORCH_CHECK(
      ceil_mode == false, "pool2d: ceil_mode is not yet implemented");

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
    const at::IntArrayRef output_padding,
    const int64_t groups) {
  at::Tensor input = inputs[0];
  at::Tensor weight = inputs[1];
  TORCH_CHECK(groups == 1, "convolution_hpu doesn't support groups");
  TORCH_CHECK(
      transposed == false, "convolution_hpu doesn't support transposition");
  TORCH_CHECK(
      std::all_of(
          output_padding.cbegin(),
          output_padding.cend(),
          [](int64_t x) { return x == 0; }),
      "convolution_hpu doesn't support output padding");
  TORCH_CHECK(
      weight.device().type() == c10::DeviceType::HABANA,
      "weight is not habana at::Tensor");
  TORCH_CHECK(weight.ndimension() == 4, "weight tensordimension count  != 4");
  TORCH_CHECK(
      weight.size(1) == input.size(1),
      "Number of input channels doesn't match weight channels");
  if (inputs.size() > 2) {
    at::Tensor bias = inputs[2];
    TORCH_CHECK(
        bias.device().type() == c10::DeviceType::HABANA,
        "bias is not habana at::Tensor");
    TORCH_CHECK(bias.dim() == 1, "bias at::Tensor idimension count  != 1");
  }

  TORCH_CHECK(
      std::all_of(
          dilation.cbegin(), dilation.cend(), [](int64_t x) { return x == 1; }),
      "convolution_hpu doesn't support dilation");
  TORCH_CHECK(
      input.device().type() == c10::DeviceType::HABANA,
      "input is not habana at::Tensor");
  TORCH_CHECK(
      stride.size() == 2, "stride size != 2 unsupported by convolution_hpu");
  TORCH_CHECK(
      padding.size() == 2, "padding size != 2 unsupported by convolution_hpu");
  TORCH_CHECK(
      input.ndimension() == 4, "input at::Tensor dimension count !=  4");

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
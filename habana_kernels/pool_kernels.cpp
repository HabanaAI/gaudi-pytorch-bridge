/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/InferSize.h>
// #include <ATen/native/Pool.h> // TODO: fix this include
#include <ATen/div_rtn.h> // TODO: remove this header after ATen/native/Pool.h is included
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#include <algorithm>
#include <iostream>

#include "conv_pool_utils.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "kernel_utils.h"

using namespace torch;

namespace { // Copy paste from ATen/native/Pool.h
template <typename dest_t, typename src_t>
static inline dest_t safe_downcast(src_t v) {
  TORCH_CHECK(
      std::numeric_limits<dest_t>::min() <= v &&
          v <= std::numeric_limits<dest_t>::max(),
      "integer out of range");

  return static_cast<dest_t>(v);
}

template <typename T>
static inline T pooling_output_shape_pad_lr(
    T inputSize,
    T kernelSize,
    T pad_l,
    T pad_r,
    T stride,
    T dilation,
    bool ceil_mode) {
  T outputSize = div_rtn<T>(
                     inputSize + pad_l + pad_r - dilation * (kernelSize - 1) -
                         1 + (ceil_mode ? stride - 1 : 0),
                     stride) +
      1;
  if (pad_l) {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((outputSize - 1) * stride >= inputSize + pad_l)
      --outputSize;
  }
  return outputSize;
}

template <typename T>
static inline T pooling_output_shape(
    T inputSize,
    T kernelSize,
    T pad,
    T stride,
    T dilation,
    bool ceil_mode) {
  return pooling_output_shape_pad_lr(
      inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode);
}

/**
 * @brief Compute shape for output tensor(s) from given input tensor shape
 *         & pooling params such as kernel, stride, pad, dilation, ceil_mode
*/
static std::vector<int64_t> compute_output_shape(
    const at::Tensor& input,
    const at::IntArrayRef kernel_size,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    bool ceil_mode) {
  const int filter_H = safe_downcast<int, int64_t>(kernel_size[0]);
  const int filter_W = kernel_size.size() == 1
      ? filter_H
      : safe_downcast<int, int64_t>(kernel_size[1]);

  const int stride_H =
      stride.empty() ? filter_H : safe_downcast<int, int64_t>(stride[0]);
  const int stride_W = stride.empty()
      ? filter_W
      : stride.size() == 1 ? stride_H : safe_downcast<int, int64_t>(stride[1]);

  const int pad_H = safe_downcast<int, int64_t>(padding[0]);
  const int pad_W =
      padding.size() == 1 ? pad_H : safe_downcast<int, int64_t>(padding[1]);

  const int dilation_H = safe_downcast<int, int64_t>(dilation[0]);
  const int dilation_W = dilation.size() == 1
      ? dilation_H
      : safe_downcast<int, int64_t>(dilation[1]);

  // input, output NCHW
  // weight KCHW, where K - output channels
  // pad, stride HW
  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  const int64_t input_H = input.size(2);
  const int64_t input_W = input.size(3);
  const int64_t output_H = pooling_output_shape<int64_t>(
      input_H, filter_H, pad_H, stride_H, dilation_H, ceil_mode);
  const int64_t output_W = pooling_output_shape<int64_t>(
      input_W, filter_W, pad_W, stride_W, dilation_W, ceil_mode);

  std::vector<int64_t> outshape{N, output_H, output_W, C};
  return outshape;
}

} // namespace

ns_SpatialReduction::Params synapse_pool_params_builder(
    const IntArrayRef& kernel_size, // HW
    const IntArrayRef& stride, // HW
    const IntArrayRef& padding, // HW
    const IntArrayRef& dilation // HW
) {
  const int64_t filter_H = kernel_size[0];
  const int64_t filter_W = kernel_size[1];
  const int64_t stride_H = stride[0];
  const int64_t stride_W = stride[1];
  const int64_t dilation_H = dilation[0];
  const int64_t dilation_W = dilation[1];

  ns_SpatialReduction::Params pool_params{};
  pool_params.kernel_w = filter_W;
  pool_params.kernel_h = filter_H;
  pool_params.stride_w = stride_W;
  pool_params.stride_h = stride_H;
  pool_params.pad_w_begin = padding[1];
  pool_params.pad_w_end = padding[1];
  pool_params.pad_h_begin = padding[0];
  pool_params.pad_h_end = padding[0];
  pool_params.dilation_w = dilation_W;
  pool_params.dilation_h = dilation_H;
  pool_params.pooling_convention = POOLING_CONVENTION_VALID;

  return pool_params;
}

/**
 * @brief MaxPool2d.with_indices_hpu (Forward Pass) implementation for Habana device
 * @param [In] Input Tensor. 4D, bf16/fp32
 * @param [In] size of the window. int64 or int64 tuple
 * @param [In] stride of the window. int64 or int64 tuple. Default: kernel_size
 * @param [In] zero padding on both sides. int64 or int64 tuple. Default: 0
 * @param [In] parameter that controls stride of elements in window. Default: 1
 * @param [In] when true use ceil instead of floor to compute output shape. Default: false
 * @param [Out] Output Tensor. 4D, bf16/fp32
 * @param [Out] Output Indices. 1D, uint8
 */
std::tuple<Tensor, Tensor> max_pool2d_with_indices_hpu(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  LOG_FUNC_BEGIN;

  habana_helpers::check_pool_params(
      input, kernel_size, stride, padding, dilation, ceil_mode);

  auto out_shape = compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode);

  //   NCHW -> NHWC
  auto input_nhwc = input.permute({0, 2, 3, 1});
  auto output_nhwc = at::empty(
      {out_shape[0], out_shape[1], out_shape[2], out_shape[3]},
      input.options());
  // NOTE: cpu and cuda implementations hold indices as kLong (int64). I am
  // using uint8 (to match TPC kernel requirement)
  auto output_idx_nhwc = at::empty(
      {out_shape[0], out_shape[1], out_shape[2], out_shape[3]},
      input.options().dtype(kByte));

  auto syn_pool_params =
      synapse_pool_params_builder(kernel_size, stride, padding, dilation);

  std::vector<const at::Tensor*> pt_inputs{&input_nhwc};
  std::vector<const at::Tensor*> pt_outputs{&output_idx_nhwc, &output_nhwc};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "maxpool_2d",
      &syn_pool_params,
      sizeof(syn_pool_params),
      SynapsePassType::FORWARD_PASS);

  //   NHWC -> NCHW
  auto output = output_nhwc.permute({0, 3, 1, 2});
  auto output_idx = output_idx_nhwc.permute({0, 3, 1, 2});
  LOG_FUNC_END;
  return {output, output_idx};
}

/**
 * @brief MaxPool2d.with_indices_hpu.out (Backward Pass) implementation for Habana device
 * @param [In/Out] Backward pass Output Tensor. 4D, bf16/fp32
 * @param [In] Backward pass Input Tensor. 4D, bf16/fp32
 * @param [In] Forward pass Input Tensor. 4D, bf16/fp32
 * @param [In] Forward pass Indices Tensor. 1D, uint8
 * @param [In] size of the window. int64 or int64 tuple
 * @param [In] stride of the window. int64 or int64 tuple. Default: kernel_size
 * @param [In] zero padding on both sides. int64 or int64 tuple. Default: 0
 * @param [In] parameter that controls stride of elements in window. Default: 1
 * @param [In] when true use ceil instead of floor to compute output shape. Default: false
 */
Tensor& max_pool2d_with_indices_backward_out_hpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  LOG_FUNC_BEGIN;
  habana_helpers::check_pool_params(
      input, kernel_size, stride, padding, dilation, ceil_mode);

  auto out_shape = compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode);

  std::vector<int64_t> expected_output_size{
      out_shape[0], out_shape[3], out_shape[1], out_shape[2]};
  TORCH_CHECK(input.sizes() == grad_input.sizes());
  TORCH_CHECK(grad_output.sizes() == indices.sizes());
  TORCH_CHECK(grad_output.sizes().vec() == expected_output_size);
  TORCH_CHECK(indices.scalar_type() == c10::ScalarType::Byte);

  //   NCHW -> NHWC
  auto grad_input_nhwc = grad_input.permute({0, 2, 3, 1});
  auto grad_output_nhwc = grad_output.permute({0, 2, 3, 1});
  auto indices_nhwc = indices.permute({0, 2, 3, 1});

  auto syn_pool_params =
      synapse_pool_params_builder(kernel_size, stride, padding, dilation);

  std::vector<const at::Tensor*> pt_inputs{&grad_output_nhwc, &indices_nhwc};
  std::vector<const at::Tensor*> pt_outputs{&grad_input_nhwc};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "maxpool_2d",
      &syn_pool_params,
      sizeof(syn_pool_params),
      SynapsePassType::BACKWARD_PASS);

  //   NHWC -> NCHW
  grad_input = grad_input_nhwc.permute({0, 3, 1, 2});

  LOG_FUNC_END;
  return grad_input;
}

/**
 * @brief MaxPool2d.with_indices_hpu (Backward Pass) implementation for Habana device
 * @param [In] Backward pass Input Tensor. 4D, bf16/fp32
 * @param [In] Forward pass Input Tensor. 4D, bf16/fp32
 * @param [In] size of the window. int64 or int64 tuple
 * @param [In] stride of the window. int64 or int64 tuple. Default: kernel_size
 * @param [In] zero padding on both sides. int64 or int64 tuple. Default: 0
 * @param [In] parameter that controls stride of elements in window. Default: 1
 * @param [In] when true use ceil instead of floor to compute output shape. Default: false
 * @param [In] Forward pass Indices Tensor. 1D, uint8
 * @param [Out] Backward pass Output Tensor. 4D, bf16/fp32
 */
Tensor max_pool2d_with_indices_backward_hpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  LOG_FUNC_BEGIN;
  // TODO: if TPC kernel write zeros than we don't have to call zero_like. Try
  // to call some function without fill
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  max_pool2d_with_indices_backward_out_hpu(
      grad_input,
      grad_output,
      input,
      indices,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
  LOG_FUNC_END;
  return grad_input;
}

/**
 * @brief AveragePool2d (Forward Pass) implementation for Habana device
 * @param [In] Input Tensor. 4D, bf16/fp32
 * @param [In] size of the window. int64 or int64 tuple
 * @param [In] stride of the window. int64 or int64 tuple. Default: kernel_size
 * @param [In] zero padding on both sides. int64 or int64 tuple. Default: 0
 * @param [In] when true use ceil instead of floor to compute output shape. Default: false
 * @param [In] when true will include zero-padding in averaging. Default: true
 * @param [In] if specified, this value will be used as divisor. Default: None
 * @param [Out] Output Tensor. 4D, bf16/fp32
 */
Tensor avg_pool2d_hpu(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  LOG_FUNC_BEGIN;

  // TODO check if TPC kernel implements count_include_pad = true or false
  TORCH_CHECK(
      count_include_pad == true,
      "avg_pool2d: count_include_pad is not yet implemented");

  TORCH_CHECK(
      !divisor_override.has_value(),
      "avgpool_2d: divisor override is not supported");

  // Dilation set to 1, since for AvgPool Pytorch API does not give dilation
  // values
  std::vector<int64_t> d{1, 1};
  IntArrayRef dilation(d.data(), d.size());

  habana_helpers::check_pool_params(
      input, kernel_size, stride, padding, dilation, ceil_mode);

  auto out_shape = compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode);

  // NCHW -> NHWC
  auto input_nhwc = input.permute({0, 2, 3, 1});
  auto output_nhwc = at::empty(
      {out_shape[0], out_shape[1], out_shape[2], out_shape[3]},
      input.options());

  // Populate pool params structure
  auto syn_pool_params =
      synapse_pool_params_builder(kernel_size, stride, padding, dilation);

  std::vector<const at::Tensor*> pt_inputs{&input_nhwc};
  std::vector<const at::Tensor*> pt_outputs{&output_nhwc};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "avg_pool_2d",
      &syn_pool_params,
      sizeof(syn_pool_params),
      SynapsePassType::FORWARD_PASS);

  //   NHWC -> NCHW
  auto output = output_nhwc.permute({0, 3, 1, 2});
  LOG_FUNC_END;
  return output;
}

/**
 * @brief AveragePool2d.out (Backward Pass) implementation for Habana device
 * @param [In/Out] Backward pass Output Tensor. 4D, bf16/fp32
 * @param [In] Backward pass Input Tensor. 4D, bf16/fp32
 * @param [In] Forward pass Input Tensor. 4D, bf16/fp32
 * @param [In] size of the window. int64 or int64 tuple
 * @param [In] stride of the window. int64 or int64 tuple. Default: kernel_size
 * @param [In] zero padding on both sides. int64 or int64 tuple. Default: 0
 * @param [In] when true use ceil instead of floor to compute output shape. Default: false
 * @param [In] when true will include zero-padding in averaging. Default: true
 * @param [In] if specified, this value will be used as divisor. Default: None
 */
Tensor& avg_pool2d_backward_out_hpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  LOG_FUNC_BEGIN;

  TORCH_CHECK(
      count_include_pad == true,
      "avg_pool2d: Pooling count_include_pad = false is not yet implemented");

  TORCH_CHECK(
      !divisor_override.has_value(),
      "avg_pool2d: divisor override is not supported");

  // Dilation set to 1, since for AvgPool Pytorch API does not give dilation
  // values
  std::vector<int64_t> d{1, 1};
  IntArrayRef dilation(d.data(), d.size());

  habana_helpers::check_pool_params(
      input, kernel_size, stride, padding, dilation, ceil_mode);

  auto out_shape = compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode);

  std::vector<int64_t> expected_output_size{
      out_shape[0], out_shape[3], out_shape[1], out_shape[2]};
  TORCH_CHECK(input.sizes() == grad_input.sizes());
  TORCH_CHECK(grad_output.sizes().vec() == expected_output_size);

  //   NCHW -> NHWC
  auto grad_input_nhwc = grad_input.permute({0, 2, 3, 1});
  auto grad_output_nhwc = grad_output.permute({0, 2, 3, 1});

  auto syn_pool_params =
      synapse_pool_params_builder(kernel_size, stride, padding, dilation);

  std::vector<const at::Tensor*> pt_inputs{&grad_output_nhwc};
  std::vector<const at::Tensor*> pt_outputs{&grad_input_nhwc};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "avg_pool_2d",
      &syn_pool_params,
      sizeof(syn_pool_params),
      SynapsePassType::BACKWARD_PASS);

  //   NHWC -> NCHW
  grad_input = grad_input_nhwc.permute({0, 3, 1, 2});

  LOG_FUNC_END;
  return grad_input;
}

/**
 * @brief AveragePool2d (Backward Pass) implementation for Habana device
 * @param [In] Backward pass Input Tensor. 4D, bf16/fp32
 * @param [In] Forward pass Input Tensor. 4D, bf16/fp32
 * @param [In] size of the window. int64 or int64 tuple
 * @param [In] stride of the window. int64 or int64 tuple. Default: kernel_size
 * @param [In] zero padding on both sides. int64 or int64 tuple. Default: 0
 * @param [In] when true use ceil instead of floor to compute output shape. Default: false
 * @param [In] when true will include zero-padding in averaging. Default: true
 * @param [In] if specified, this value will be used as divisor. Default: None
 * @param [Out] Backward pass Output Tensor. 4D, bf16/fp32
 */
Tensor avg_pool2d_backward_hpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  LOG_FUNC_BEGIN;
  // TODO: if TPC kernel write zeros than we don't have to call zero_like. Try
  // to call some function without fill
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  avg_pool2d_backward_out_hpu(
      grad_input,
      grad_output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
  LOG_FUNC_END;
  return grad_input;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride = [], int[2] padding = 0, int[2] dilation = 1, bool ceil_mode = False) ->(Tensor, Tensor)")
                .impl_unboxedOnlyKernel<
                    decltype(max_pool2d_with_indices_hpu),
                    &max_pool2d_with_indices_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(max_pool2d_with_indices_backward_hpu),
                    &max_pool2d_with_indices_backward_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(max_pool2d_with_indices_backward_out_hpu),
                    &max_pool2d_with_indices_backward_out_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(avg_pool2d_hpu),
                    &avg_pool2d_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(avg_pool2d_backward_hpu),
                    &avg_pool2d_backward_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
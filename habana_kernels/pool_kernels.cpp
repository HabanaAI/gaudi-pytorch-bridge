#include <ATen/InferSize.h>
// #include <ATen/native/Pool.h> // TODO: fix this include
#include <ATen/div_rtn.h> // TODO: remove this header after ATen/native/Pool.h is included
#include <torch/script.h>
#include <tpc_kernels/include/perf_lib_layer_params.h>
#include <algorithm>
#include <iostream>

#include "conv_pool_utils.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
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

  TORCH_CHECK(padding[0] == 0);
  TORCH_CHECK(padding[1] == 0);

  ns_SpatialReduction::Params pool_params{};
  pool_params.kernel_w = filter_W;
  pool_params.kernel_h = filter_H;
  pool_params.stride_w = stride_W;
  pool_params.stride_h = stride_H;
  // TODO: add padding support
  pool_params.pad_w_begin = 0;
  pool_params.pad_w_end = 0;
  pool_params.pad_h_begin = 0;
  pool_params.pad_h_end = 0;
  pool_params.dilation_w = dilation_W;
  pool_params.dilation_h = dilation_H;
  pool_params.pooling_convention = POOLING_CONVENTION_VALID;

  return pool_params;
}

void synapse_pool2d_generic_impl(
    std::vector<const Tensor*> pt_outputs, // NHWC
    std::vector<const Tensor*> pt_inputs, // NHWC
    IntArrayRef kernel_size, // HW
    IntArrayRef stride, // HW
    IntArrayRef padding, // HW
    IntArrayRef dilation, // HW
    bool forward_pass) {
  // TODO: implement support for padding
  const auto device_id = pt_inputs[0]->device().index();
  // graph_handle scope
  synGraphHandle graph_handle;
  TORCH_HABANA_CHECK(
      synGraphCreate(&graph_handle, synDeviceType::synDeviceGaudi),
      "synGraphCreate failed");
  { // tensors scope
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_outputs;
    std::vector<synTensor> syn_inputs, syn_outputs;

    std::tie(syn_helper_inputs, syn_inputs) =
        habana_helpers::create_tensors(pt_inputs, graph_handle, true);
    std::tie(syn_helper_outputs, syn_outputs) =
        habana_helpers::create_tensors(pt_outputs, graph_handle, true);

    {
      const std::string node_type = "maxpool_2d_" +
          std::string(forward_pass ? "fwd_" : "bwd_") +
          habana_helpers::name_suffix_from_type(pt_inputs[0]->scalar_type());
      { // add node
        auto syn_pool_params =
            synapse_pool_params_builder(kernel_size, stride, padding, dilation);

        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                syn_inputs.data(),
                syn_outputs.data(),
                syn_inputs.size(),
                syn_outputs.size(),
                &syn_pool_params,
                sizeof(syn_pool_params),
                node_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");
      }

      habana_helpers::compile_and_run(
          node_type,
          graph_handle,
          habana_helpers::names(syn_helper_inputs),
          habana_helpers::names(syn_helper_outputs),
          habana_helpers::extract_data_ptrs(pt_inputs),
          habana_helpers::extract_data_ptrs(pt_outputs),
          device_id);
    }
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}

std::tuple<Tensor, Tensor> max_pool2d_with_indices_hpu(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  LOG_FUNC_BEGIN;

  // TODO:: add support for ceil mode
  TORCH_CHECK(ceil_mode == false, "Pooling ceil_mode is not yet implemented");
  habana_helpers::check_pool_params(input, stride, padding, dilation);

  // input, output NCHW
  // weight KCHW, where K - output channels
  // pad, stride HW
  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  const int64_t input_H = input.size(2);
  const int64_t input_W = input.size(3);
  const int64_t filter_H = kernel_size[0];
  const int64_t filter_W = kernel_size[1];
  const int64_t stride_H = stride[0];
  const int64_t stride_W = stride[1];
  const int64_t pad_H = padding[0];
  const int64_t pad_W = padding[1];
  const auto output_H = habana_helpers::compute_output_size(
      input_H, pad_H, filter_H, stride_H, ceil_mode);
  const auto output_W = habana_helpers::compute_output_size(
      input_W, pad_W, filter_W, stride_W, ceil_mode);

  //   NCHW -> NHWC
  auto input_nhwc = input.permute({0, 2, 3, 1});
  auto output_nhwc = at::empty({N, output_H, output_W, C}, input.options());
  // NOTE: cpu and cuda implementations hold indices as kLong (int64). I am
  // using uint8
  auto output_idx_nhwc =
      at::empty({N, output_H, output_W, C}, input.options().dtype(kByte));

  synapse_pool2d_generic_impl(
      {&output_idx_nhwc, &output_nhwc},
      {&input_nhwc},
      kernel_size,
      stride,
      padding,
      dilation,
      true);

  //   NHWC -> NCHW
  auto output = output_nhwc.permute({0, 3, 1, 2});
  auto output_idx = output_idx_nhwc.permute({0, 3, 1, 2});
  LOG_FUNC_END;
  return {output, output_idx};
}

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
  // TODO: merge pt constriants check with check_pool_params function
  TORCH_CHECK(!ceil_mode, "Pooling ceil_mode is not yet implemented");
  habana_helpers::check_pool_params(input, stride, padding, dilation);

  // ############### Copy paste check from PT code
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int filer_H = safe_downcast<int, int64_t>(kernel_size[0]);
  const int filer_W = kernel_size.size() == 1
      ? filer_H
      : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we
  // accept empty stride for this case
  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int stride_H =
      stride.empty() ? filer_H : safe_downcast<int, int64_t>(stride[0]);
  const int stride_W = stride.empty()
      ? filer_W
      : stride.size() == 1 ? stride_H : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int pad_H = safe_downcast<int, int64_t>(padding[0]);
  const int pad_W =
      padding.size() == 1 ? pad_H : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilation_H = safe_downcast<int, int64_t>(dilation[0]);
  const int dilation_W = dilation.size() == 1
      ? dilation_H
      : safe_downcast<int, int64_t>(dilation[1]);
  // ############### End of copy paste check from PT code

  const int64_t N = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t C = input.size(-3);
  const int64_t input_H = input.size(-2);
  const int64_t input_W = input.size(-1);

  // TODO: reuse pooling_output_shape for pool fwd and conv if possible
  const int64_t output_H = pooling_output_shape<int64_t>(
      input_H, filer_H, pad_H, stride_H, dilation_H, ceil_mode);
  const int64_t output_W = pooling_output_shape<int64_t>(
      input_W, filer_W, pad_W, stride_W, dilation_W, ceil_mode);

  std::vector<int64_t> expected_output_size{N, C, output_H, output_W};
  TORCH_CHECK(input.sizes() == grad_input.sizes());
  TORCH_CHECK(grad_output.sizes() == indices.sizes());
  TORCH_CHECK(grad_output.sizes().vec() == expected_output_size);
  TORCH_CHECK(indices.scalar_type() == c10::ScalarType::Byte);

  //   NCHW -> NHWC
  auto grad_input_nhwc = grad_input.permute({0, 2, 3, 1});
  auto grad_output_nhwc = grad_output.permute({0, 2, 3, 1});
  auto input_nhwc = input.permute({0, 2, 3, 1});
  auto indices_nhwc = indices.permute({0, 2, 3, 1});

  synapse_pool2d_generic_impl(
      {&grad_input_nhwc},
      {&grad_output_nhwc, &indices_nhwc},
      kernel_size,
      stride,
      padding,
      dilation,
      false);

  //   NHWC -> NCHW
  grad_input = grad_input_nhwc.permute({0, 3, 1, 2});

  LOG_FUNC_END;
  return grad_input;
}

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
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));

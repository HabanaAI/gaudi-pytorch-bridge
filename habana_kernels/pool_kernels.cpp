#include <ATen/InferSize.h>
#include <torch/script.h>
#include <tpc_kernels/include/perf_lib_layer_params.h>
#include <algorithm>
#include <iostream>

#include "conv_pool_utils.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_helpers/tensor_utils.h"
#include "kernel_utils.h"

using namespace torch;

ns_SpatialReduction::Params synapse_pool_params_builder(
    const IntArrayRef& kernel_size,
    const IntArrayRef& stride,
    const IntArrayRef& padding,
    const IntArrayRef& dilation) {
  const int64_t filter_H = kernel_size[0];
  const int64_t filter_W = kernel_size[1];
  const int64_t stride_H = stride[0];
  const int64_t stride_W = stride[1];
  const int64_t pad_H = padding[0];
  const int64_t pad_W = padding[1];
  const int64_t dilation_H = dilation[0];
  const int64_t dilation_W = dilation[1];

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

void synapse_pool(
    const Tensor& output_idx,
    const Tensor& output,
    const Tensor& input,
    const IntArrayRef& kernel_size,
    const IntArrayRef& stride,
    const IntArrayRef& padding,
    const IntArrayRef& dilation) {
  const auto device_id = input.device().index();
  // graph_handle scope
  synGraphHandle graph_handle;
  TORCH_HABANA_CHECK(
      synGraphCreate(&graph_handle, synDeviceType::synDeviceGaudi),
      "synGraphCreate failed");
  { // tensors scope
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_outputs;
    std::vector<synTensor> syn_inputs, syn_outputs;

    std::tie(syn_helper_inputs, syn_inputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&input},
        {"input"},
        graph_handle,
        {true});
    std::tie(syn_helper_outputs, syn_outputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&output_idx, &output},
        {"output_idx", "output"},
        graph_handle,
        {true, true});

    { // dimshuffled tensors scope
#if TRANSPOSE_IMPLEMENTED
      // input, filter Note: I will use original bias
      std::vector<synapse_helpers::tensor> syn_tmp_helper_inputs;
      std::vector<synapse_helpers::tensor> syn_tmp_helper_outputs;
      const std::vector<std::string> input_tmp_names{"input_tmp"};
      const std::vector<std::string> output_tmp_names{"output_tmp",
                                                      "output_idx_tmp"};

      syn_tmp_helper_inputs.push_back(
          synapse_helpers::tensor_builder::create_tensor(
              device_id,
              synDataType::syn_type_float,
              input.nbytes(),
              input.sizes().size(),
              habana_helpers::hack_pytorch_nhwc_shapes(
                  input.sizes(), TRANSPOSE_IMPLEMENTED == true),
              input_tmp_names[0],
              graph_handle,
              false));
      syn_tmp_helper_outputs.push_back(
          synapse_helpers::tensor_builder::create_tensor(
              device_id,
              synDataType::syn_type_float,
              output.nbytes(),
              output.sizes().size(),
              habana_helpers::hack_pytorch_nhwc_shapes(
                  output.sizes(), TRANSPOSE_IMPLEMENTED == true),
              output_tmp_names[0],
              graph_handle,
              false));
      syn_tmp_helper_outputs.push_back(
          synapse_helpers::tensor_builder::create_tensor(
              device_id,
              synDataType::syn_type_uint8,
              output_idx.nbytes(),
              output_idx.sizes().size(),
              habana_helpers::hack_pytorch_nhwc_shapes(
                  output_idx.sizes(), TRANSPOSE_IMPLEMENTED == true),
              output_tmp_names[1],
              graph_handle,
              false));

      std::vector<synTensor> syn_tmp_inputs(syn_tmp_helper_inputs.size());
      std::vector<synTensor> syn_tmp_outputs(syn_tmp_helper_outputs.size());
      std::transform(
          syn_tmp_helper_inputs.begin(),
          syn_tmp_helper_inputs.end(),
          syn_tmp_inputs.begin(),
          [](auto& x) { return x.get(); });
      std::transform(
          syn_tmp_helper_outputs.begin(),
          syn_tmp_helper_outputs.end(),
          syn_tmp_outputs.begin(),
          [](auto& x) { return x.get(); });

      const std::string transpose_node_type = "transpose";

      { // add input transpositions
        // Transpose(0,2,3,1), transform NCHW data format to NHWC
        synTransposeParams params_nchw_to_nhwc;
        {
          params_nchw_to_nhwc.tensorDim = 4;
          params_nchw_to_nhwc.permutation[0] = TransposePermutationDim(0);
          params_nchw_to_nhwc.permutation[1] = TransposePermutationDim(2);
          params_nchw_to_nhwc.permutation[2] = TransposePermutationDim(3);
          params_nchw_to_nhwc.permutation[3] = TransposePermutationDim(1);
        }

        // dimshuffle input
        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                &syn_inputs[0],
                &syn_tmp_inputs[0],
                1,
                1,
                &params_nchw_to_nhwc,
                sizeof(params_nchw_to_nhwc),
                transpose_node_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");
      }
#endif
      const std::string pool_node_type = "maxpool_2d_fwd_" +
          habana_helpers::name_suffix_from_type(input.scalar_type());
      { // add pool node
        // TODO: support pytorch layouts, uncomment when it is supported and
        // remove transpositions
        //   char const* pool2D_in_layouts[]{"WHCN"};
        //   char const* pool2D_out_layouts[]{"WHCN", "WHCN"};

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
                pool_node_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");
      }
#if TRANSPOSE_IMPLEMENTED
      { // add output transpose node
        synTransposeParams params_nhwc_to_nchw;
        {
          params_nhwc_to_nchw.tensorDim = 4;
          params_nhwc_to_nchw.permutation[0] = TransposePermutationDim(0);
          params_nhwc_to_nchw.permutation[1] = TransposePermutationDim(3);
          params_nhwc_to_nchw.permutation[2] = TransposePermutationDim(2);
          params_nhwc_to_nchw.permutation[3] = TransposePermutationDim(1);
        }
        // dimshuffle output
        for (int i = 0; i < syn_outputs.size(); ++i)
          TORCH_HABANA_CHECK(
              synNodeCreate(
                  graph_handle,
                  &syn_tmp_outputs[i],
                  &syn_outputs[i],
                  1,
                  1,
                  &params_nhwc_to_nchw,
                  sizeof(params_nhwc_to_nchw),
                  transpose_node_type.c_str(),
                  "",
                  nullptr,
                  nullptr),
              "synNodeCreate failed");
      }
#endif

      habana_helpers::compile_and_run(
          pool_node_type,
          graph_handle,
          habana_helpers::names(syn_helper_inputs),
          habana_helpers::names(syn_helper_outputs),
          {input.data_ptr()},
          {output_idx.data_ptr(), output.data_ptr()},
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
  std::cout << "max_pool2d_with_indices_hpu called\n"; // TODO: remove

  // TODO:: add support for ceil mode
  TORCH_CHECK(ceil_mode == false, "Pooling ceil_mode is not yet implemented");
  habana_helpers::check_pool_params(
      input, kernel_size, stride, padding, dilation);

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

  synapse_pool(
      output_idx_nhwc,
      output_nhwc,
      input_nhwc,
      kernel_size,
      stride,
      padding,
      dilation);

  //   NHWC -> NCHW
  auto output = output_nhwc.permute({0, 3, 1, 2});
  auto output_idx = output_idx_nhwc.permute({0, 3, 1, 2});

  return {output, output_idx};
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride = [], int[2] padding = 0, int[2] dilation = 1, bool ceil_mode = False) ->(Tensor, Tensor) ")
        .impl_unboxedOnlyKernel<
            decltype(max_pool2d_with_indices_hpu),
            &max_pool2d_with_indices_hpu>(TensorTypeId::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));

#include <ATen/InferSize.h>
#include <synapse/include/synapse_api.h>
#include <torch/script.h>
#include <iostream>
#include <string>

#include "conv_pool_utils.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_helpers/tensor_utils.h"
#include "kernel_utils.h"

using namespace torch;

synConvolutionParams synapse_conv_params_builder(
    const IntArrayRef& input, // NHWC
    const IntArrayRef& weight, // HWCK
    const IntArrayRef& stride,
    const IntArrayRef& padding,
    const IntArrayRef& dilation) {
  const int64_t C = input[3];
  const int64_t input_H = input[1];
  const int64_t input_W = input[2];
  const int64_t K = weight[3];
  const int64_t filter_H = weight[0];
  const int64_t filter_W = weight[1];
  const int64_t stride_H = stride[0];
  const int64_t stride_W = stride[1];
  const int64_t pad_H = padding[0];
  const int64_t pad_W = padding[1];
  const int64_t dilation_H = dilation[0];
  const int64_t dilation_W = dilation[1];
  // TODO: calculate paddings

  synConvolutionParams syn_conv_params{};
  syn_conv_params.dH = stride_H;
  syn_conv_params.dW = stride_W;
  syn_conv_params.kH = filter_H;
  syn_conv_params.kW = filter_W;
  syn_conv_params.dilH = dilation_H;
  syn_conv_params.dilW = dilation_W;
  syn_conv_params.setPadT(0);
  syn_conv_params.setPadB(0);
  syn_conv_params.setPadL(0);
  syn_conv_params.setPadR(0);

  return syn_conv_params;
}

void synapse_convolution(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
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
    const std::vector<std::string> input_names{"input", "filter", "bias"};
    const std::vector<std::string> output_names{"output"};

    std::vector<synapse_helpers::tensor> syn_helper_inputs{};
    syn_helper_inputs.push_back(
        habana_helpers::create_tensor(input, input_names[0], true));
    syn_helper_inputs.push_back(
        habana_helpers::create_tensor(weight, input_names[1], true));
    syn_helper_inputs.push_back(
        habana_helpers::create_tensor(bias, input_names[2], true));

    std::vector<synapse_helpers::tensor> syn_helper_outputs{};
    syn_helper_outputs.push_back(
        habana_helpers::create_tensor(output, output_names[0], true));

    // workaround for missing synapse_helpers::graph support
    std::vector<synTensor> syn_inputs(
        syn_helper_inputs.size()); // input, filter, bias
    std::vector<synTensor> syn_outputs(syn_helper_outputs.size()); //  output
    std::transform(
        syn_helper_inputs.begin(),
        syn_helper_inputs.end(),
        syn_inputs.begin(),
        [](auto& x) { return x.get(); });
    std::transform(
        syn_helper_outputs.begin(),
        syn_helper_outputs.end(),
        syn_outputs.begin(),
        [](auto& x) { return x.get(); });

    { // dimshuffled tensors scope
#if TRANSPOSE_IMPLEMENTED
      // input, filter Note: I will use original bias
      std::vector<synapse_helpers::tensor> syn_tmp_helper_inputs;
      std::vector<synapse_helpers::tensor> syn_tmp_helper_outputs;
      const std::vector<std::string> input_tmp_names{"input_tmp", "filter_tmp"};
      const std::vector<std::string> output_tmp_names{"output_tmp"};

      syn_tmp_helper_inputs.push_back(
          synapse_helpers::tensor_builder::create_tensor(
              device_id,
              synDataType::syn_type_float,
              input.nbytes(),
              input.sizes().size(),
              habana_helpers::hack_pytorch_nhwc_shapes(
                  input.sizes(), TRANSPOSE_IMPLEMENTED == true),
              input_tmp_names[0],
              false));
      syn_tmp_helper_inputs.push_back(
          synapse_helpers::tensor_builder::create_tensor(
              device_id,
              synDataType::syn_type_float,
              weight.nbytes(),
              weight.sizes().size(),
              habana_helpers::hack_pytorch_nhwc_shapes(
                  weight.sizes(), TRANSPOSE_IMPLEMENTED == true),
              input_tmp_names[1],
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

      { // add input and weight transpositions
        // Transpose(0,2,3,1), transform NCHW data format to NHWC
        synTransposeParams params_NCHW_to_NHWC;
        {
          params_NCHW_to_NHWC.tensorDim = 4;
          params_NCHW_to_NHWC.permutation[0] = TransposePermutationDim(0);
          params_NCHW_to_NHWC.permutation[1] = TransposePermutationDim(2);
          params_NCHW_to_NHWC.permutation[2] = TransposePermutationDim(3);
          params_NCHW_to_NHWC.permutation[3] = TransposePermutationDim(1);
        }

        synTransposeParams params_KCHW_to_HWCK;
        {
          params_KCHW_to_HWCK.tensorDim = 4;
          params_KCHW_to_HWCK.permutation[0] = TransposePermutationDim(2);
          params_KCHW_to_HWCK.permutation[1] = TransposePermutationDim(3);
          params_KCHW_to_HWCK.permutation[2] = TransposePermutationDim(1);
          params_KCHW_to_HWCK.permutation[3] = TransposePermutationDim(0);
        }
        // dimshuffle input
        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                &syn_inputs[0],
                &syn_tmp_inputs[0],
                1,
                1,
                &params_NCHW_to_NHWC,
                sizeof(params_NCHW_to_NHWC),
                transpose_node_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");

        // dimshuffle weights
        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                &syn_inputs[1],
                &syn_tmp_inputs[1],
                1,
                1,
                &params_KCHW_to_HWCK,
                sizeof(params_KCHW_to_HWCK),
                transpose_node_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");
      }
#endif
      const std::string conv_node_type = "spatial_convolution";
      { // add conv node
        // TODO: support pytorch layouts, uncomment when it is supported and
        // remove transpositions
        //   char const* conv2D_in_layouts[]{"WHCN", "RSCK", "", "WHCN"};
        //   char const* conv2D_out_layouts[]{"WHCN"};

        synConvolutionParams syn_conv_params = synapse_conv_params_builder(
            input.sizes(), weight.sizes(), stride, padding, dilation);

        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                syn_inputs.data(),
                syn_outputs.data(),
                syn_inputs.size(),
                syn_outputs.size(),
                &syn_conv_params,
                sizeof(syn_conv_params),
                conv_node_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");
      }
#if TRANSPOSE_IMPLEMENTED
      { // add output transpose node
        synTransposeParams params_NHWC_to_NCHW;
        {
          params_NHWC_to_NCHW.tensorDim = 4;
          params_NHWC_to_NCHW.permutation[0] = TransposePermutationDim(0);
          params_NHWC_to_NCHW.permutation[1] = TransposePermutationDim(3);
          params_NHWC_to_NCHW.permutation[2] = TransposePermutationDim(1);
          params_NHWC_to_NCHW.permutation[3] = TransposePermutationDim(2);
        }
        // dimshuffle output
        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                &syn_tmp_outputs[0],
                &syn_outputs[0],
                1,
                1,
                &params_NHWC_to_NCHW,
                sizeof(params_NHWC_to_NCHW),
                transpose_node_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");
      }
#endif

      habana_helpers::compile_and_run(
          conv_node_type,
          graph_handle,
          input_names,
          output_names,
          {input.data_ptr(), weight.data_ptr(), bias.data_ptr()},
          {output.data_ptr()},
          device_id);
    }
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}

Tensor habana_convolution(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups) {
  std::cout << "habana_convolution called\n"; // TODO: remove

  habana_helpers::check_convolution_params(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups);

  // input, output NCHW
  // weight KCHW, where K - output channels
  // pad, stride HW
  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  const int64_t input_H = input.size(2);
  const int64_t input_W = input.size(3);
  const int64_t K = weight.size(0);
  const int64_t filter_H = weight.size(2);
  const int64_t filter_W = weight.size(3);
  const int64_t stride_H = stride[0];
  const int64_t stride_W = stride[1];
  const int64_t pad_H = padding[0];
  const int64_t pad_W = padding[1];
  const auto output_H = habana_helpers::compute_output_size(
      input_H, pad_H, filter_H, stride_H, false);
  const auto output_W = habana_helpers::compute_output_size(
      input_W, pad_W, filter_W, stride_W, false);

  const auto output_tensor_options = TensorOptions()
                                         .dtype(input.dtype())
                                         .device(input.device())
                                         .layout(input.layout());

  auto output_NHWC =
      at::empty({N, output_H, output_W, K}, output_tensor_options);

  // Create dimshuffled inputs and outputs to match synapse data layout
  //   NCHW -> NHWC
  auto input_NHWC = input.permute({0, 2, 3, 1});
  //   KCHW -> HWCK
  auto weight_HWCK = weight.permute({2, 3, 1, 0});

  synapse_convolution(
      output_NHWC, input_NHWC, weight_HWCK, bias, stride, padding, dilation);

  //   NHWC -> NCHW
  auto output = output_NHWC.permute({0, 3, 1, 2});
  return output;
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor")
        .impl_unboxedOnlyKernel<
            decltype(habana_convolution),
            &habana_convolution>(TensorTypeId::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));

#include <ATen/InferSize.h>
#include <torch/script.h>
#include <algorithm>
#include <iostream>
#include <string>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/fake_tensor_builder.h"
#include "kernel_utils.h"
#include "synapse/include/synapse_api.h"

#define TRANSPOSE_IMPLEMENTED false

using namespace torch;

int64_t compute_output_size(
    int64_t input,
    int64_t pad,
    int64_t filter,
    int64_t stride) {
  return (input + 2 * pad - filter) / stride + 1;
}

IntArrayRef NCHW_to_NHWC_shape(const IntArrayRef& shape) {
  return {shape[0], shape[2], shape[3], shape[1]};
}

synConvolutionParams synapse_conv_params_builder(
    const IntArrayRef& input, // NCHW
    const IntArrayRef& weight, // HWCK
    const IntArrayRef& stride,
    const IntArrayRef& padding,
    const IntArrayRef& dilation) {
  const int64_t C = input[1];
  const int64_t input_H = input[2];
  const int64_t input_W = input[3];
  const int64_t K = weight[0];
  const int64_t filter_H = weight[2];
  const int64_t filter_W = weight[3];
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

    auto hack_pytorch_nhwc_shapes =
        [](const IntArrayRef& sizes, bool hack_shapes) -> std::vector<int64_t> {
      if (hack_shapes)
        // pytorch data format is NCHW, synapse require NHWC but it is reading
        // backwards
        return std::vector<int64_t>{sizes[0], sizes[2], sizes[3], sizes[1]};
      else
        return sizes.vec();
    };

    auto hack_pytorch_kchw_shapes =
        [](const IntArrayRef& sizes, bool hack_shapes) -> std::vector<int64_t> {
      if (hack_shapes)
        // KCHW -> HWCK
        return std::vector<int64_t>{sizes[2], sizes[3], sizes[1], sizes[0]};
      else
        return sizes.vec();
    };

    std::vector<synapse_helpers::tensor> syn_helper_inputs{};
    syn_helper_inputs.push_back(synapse_helpers::tensor_builder::create_tensor(
        device_id,
        synDataType::syn_type_float,
        input.nbytes(),
        input.sizes().size(),
        hack_pytorch_nhwc_shapes(input.sizes(), TRANSPOSE_IMPLEMENTED == false),
        input_names[0],
        true));
    syn_helper_inputs.push_back(synapse_helpers::tensor_builder::create_tensor(
        device_id,
        synDataType::syn_type_float,
        weight.nbytes(),
        weight.sizes().size(),
        hack_pytorch_kchw_shapes(
            weight.sizes(), TRANSPOSE_IMPLEMENTED == false),
        input_names[1],
        true));
    syn_helper_inputs.push_back(synapse_helpers::tensor_builder::create_tensor(
        device_id,
        synDataType::syn_type_float,
        bias.nbytes(),
        bias.sizes().size(),
        bias.sizes(),
        input_names[2],
        true));
    std::vector<synapse_helpers::tensor> syn_helper_outputs{};
    syn_helper_outputs.push_back(synapse_helpers::tensor_builder::create_tensor(
        device_id,
        synDataType::syn_type_float,
        output.nbytes(),
        output.sizes().size(),
        hack_pytorch_nhwc_shapes(
            output.sizes(), TRANSPOSE_IMPLEMENTED == false),
        output_names[0],
        true));

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
              hack_pytorch_nhwc_shapes(
                  input.sizes(), TRANSPOSE_IMPLEMENTED == true),
              input_tmp_names[0],
              false));
      syn_tmp_helper_inputs.push_back(
          synapse_helpers::tensor_builder::create_tensor(
              device_id,
              synDataType::syn_type_float,
              weight.nbytes(),
              weight.sizes().size(),
              hack_pytorch_nhwc_shapes(
                  weight.sizes(), TRANSPOSE_IMPLEMENTED == true),
              input_tmp_names[1],
              false));
      syn_tmp_helper_outputs.push_back(
          synapse_helpers::tensor_builder::create_tensor(
              device_id,
              synDataType::syn_type_float,
              output.nbytes(),
              output.sizes().size(),
              hack_pytorch_nhwc_shapes(
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
                transpose_node_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");
      }
#endif
      const std::string conv_node_type = "spatial_convolution";
      { // add conv node
        char const* conv2D_in_layouts[]{"CWHN", "KCSR", "", "CWHN"};
        char const* conv2D_out_layouts[]{"CWHN"};
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
                conv_node_type.c_str(),
                "",
                conv2D_in_layouts,
                conv2D_out_layouts),
            "synNodeCreate failed");
      }
#if TRANSPOSE_IMPLEMENTED
      { // add output transpose node
        synTransposeParams params_NHWC_to_NCHW;
        {
          params_NHWC_to_NCHW.tensorDim = 4;
          params_NHWC_to_NCHW.permutation[0] = TransposePermutationDim(0);
          params_NHWC_to_NCHW.permutation[1] = TransposePermutationDim(3);
          params_NHWC_to_NCHW.permutation[2] = TransposePermutationDim(2);
          params_NHWC_to_NCHW.permutation[3] = TransposePermutationDim(1);
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
                transpose_node_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");
      }
#endif

      { // graph compilation, workspace buffer and topology buffer
        // allocation
        synRecipeHandle recipe_handle;
        const auto recipe_name =
            habana_helpers::unique_recipe_name_generator(conv_node_type);
        TORCH_HABANA_CHECK(
            synGraphCompile(
                &recipe_handle,
                graph_handle,
                recipe_name.c_str(),
                nullptr,
                0,
                0),
            "synGraphCompile failed");

        uint64_t topology_size_bytes, workspace_size_bytes;
        TORCH_HABANA_CHECK(
            synRecipeGetSize(&topology_size_bytes, recipe_handle),
            "synRecipeGetSize failed");
        TORCH_HABANA_CHECK(
            synWorkspaceGetSize(&workspace_size_bytes, recipe_handle),
            "synWorkspaceGetSize failed");

        auto hpu_raii_allocator = at::habana::getHABANADeviceAllocator();

        at::DataPtr topology_buffer =
            hpu_raii_allocator->allocate(topology_size_bytes);
        at::DataPtr workspace_buffer;
        if (workspace_size_bytes)
          workspace_buffer = hpu_raii_allocator->allocate(workspace_size_bytes);

        { // recipe upload scope
          const synRecipeInfo recipe_info{
              recipe_name.c_str(),
              reinterpret_cast<uint64_t>(topology_buffer.get())};
          TORCH_HABANA_CHECK(
              synRecipeUpload(recipe_handle, &recipe_info, device_id),
              "synRecipeUpload failed");
          { // stream handle scope
            synStreamHandle stream_handle;
            TORCH_HABANA_CHECK(
                synStreamCreate(&stream_handle, device_id, 0),
                "synStreamCreate failed");

            std::vector<synLaunchTensorInfo> syn_inputs_info{
                {input_names[0].c_str(),
                 reinterpret_cast<uint64_t>(input.data_ptr())},
                {input_names[1].c_str(),
                 reinterpret_cast<uint64_t>(weight.data_ptr())},
                {input_names[2].c_str(),
                 reinterpret_cast<uint64_t>(bias.data_ptr())}};
            std::vector<synLaunchTensorInfo> syn_outputs_info{
                {output_names[0].c_str(),
                 reinterpret_cast<uint64_t>(output.data_ptr())}};

            TORCH_HABANA_CHECK(
                synLaunch(
                    stream_handle,
                    syn_inputs_info.data(),
                    syn_inputs_info.size(),
                    syn_outputs_info.data(),
                    syn_outputs_info.size(),
                    reinterpret_cast<uint64_t>(workspace_buffer.get()),
                    &recipe_info),
                "synLaunch failed");
            TORCH_HABANA_CHECK(
                synStreamSynchronize(stream_handle),
                "synStreamSynchronize failed");

            TORCH_HABANA_CHECK(
                synStreamDestroy(stream_handle), "synStreamDestroy failed");
          }
          TORCH_HABANA_CHECK(
              synRecipeUnload(recipe_handle, &recipe_info, device_id),
              "synRecipeUnload failed");
        }
      }
    }
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}

void check_convolution_params(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool transposed,
    const IntArrayRef output_padding,
    const int64_t groups) {
  TORCH_CHECK(groups == 1, "habana_convolution doesn't support groups");
  TORCH_CHECK(
      transposed == false, "habana_convolution doesn't support transposition");
  TORCH_CHECK(
      std::all_of(
          dilation.cbegin(), dilation.cend(), [](int64_t x) { return x == 1; }),
      "habana_convolution doesn't support dilation");
  TORCH_CHECK(
      std::all_of(
          padding.cbegin(), padding.cend(), [](int64_t x) { return x == 0; }),
      "habana_convolution doesn't support input padding");
  TORCH_CHECK(
      std::all_of(
          output_padding.cbegin(),
          output_padding.cend(),
          [](int64_t x) { return x == 0; }),
      "habana_convolution doesn't support output padding");
  TORCH_CHECK(
      input.device().type() == c10::DeviceType::HABANA,
      "input is not habana tensor");
  TORCH_CHECK(
      weight.device().type() == c10::DeviceType::HABANA,
      "weight is not habana tensor");
  TORCH_CHECK(
      bias.device().type() == c10::DeviceType::HABANA,
      "bias is not habana tensor");
  TORCH_CHECK(
      stride.size() == 2, "stride size != 2 unsupported by habana_convolution");
  TORCH_CHECK(
      padding.size() == 2,
      "padding size != 2 unsupported by habana_convolution");
  TORCH_CHECK(
      input.scalar_type() == c10::ScalarType::Float,
      "input tensor is not float32");
  TORCH_CHECK(
      weight.scalar_type() == c10::ScalarType::Float,
      "weight tensor is not float32");
  TORCH_CHECK(
      bias.scalar_type() == c10::ScalarType::Float,
      "bias tensor is not float32");
  TORCH_CHECK(input.ndimension() == 4, "input tensor dimension count != 4");
  TORCH_CHECK(weight.ndimension() == 4, "weight tensordimension count != 4");
  TORCH_CHECK(bias.ndimension() == 1, "bias tensor idimension count != 1");
  TORCH_CHECK(
      weight.size(1) == input.size(1),
      "Number of input channels doesn't match weight channels");
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

  check_convolution_params(
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
  const auto output_H = compute_output_size(input_H, pad_H, filter_H, stride_H);
  const auto output_W = compute_output_size(input_W, pad_W, filter_W, stride_W);
  std::cout << "input_size N " << N << ", C " << C << ", H " << input_H
            << ", W " << input_W << '\n'; // TODO: remove
  const auto output_tensor_options = TensorOptions()
                                         .dtype(input.dtype())
                                         .device(input.device())
                                         .layout(input.layout());
  auto output = at::empty({N, K, output_H, output_W}, output_tensor_options);
  synapse_convolution(output, input, weight, bias, stride, padding, dilation);

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

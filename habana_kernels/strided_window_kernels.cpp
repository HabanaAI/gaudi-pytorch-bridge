#include <algorithm>
#include <iostream>
#include <string>

#include <ATen/InferSize.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "synapse/include/synapse_api.h"

// #define TRANSPOSE_IMPLEMENTED

using namespace torch;

std::string unique_recipe_name_generator(std::string recipe_name) {
  static std::unordered_map<std::string, unsigned> map;
  return recipe_name + std::to_string(map[recipe_name]++);
}

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

synTensorDescriptorTr synapse_tensor_descriptor_builder(
    const IntArrayRef& shape,
    const synDataType dtype,
    const std::string& name,
    const bool persistent) {
  synTensorDescriptorTr descriptor;
  descriptor.m_dataType = dtype;
  descriptor.m_dims = shape.size();

  TORCH_CHECK(
      shape.size() <= SYN_MAX_TENSOR_DIM,
      name,
      " tensor has more than ",
      SYN_MAX_TENSOR_DIM,
      " dimensions");
  // write NHWC as CWHN and write 0 at the end
  for (int i = 0; i < shape.size(); ++i)
    descriptor.m_sizes[i] = shape[shape.size() - i - 1];
  for (int i = shape.size(); i < SYN_MAX_TENSOR_DIM; ++i)
    descriptor.m_sizes[i] = 0;

  //   descriptor.m_strides[SYN_MAX_TENSOR_DIM]; // TODO: not needed?
  descriptor.m_name = name.c_str(); // TODO: we take only pointer, so make sure
                                    // name object will be alive
  descriptor.m_deviceMemAddress = 0; // It will be patched during runtime
  descriptor.m_isOutput = persistent;
  descriptor.m_isPersistent = persistent;

  return descriptor;
}

synConvolutionParams synapse_conv_params_builder(
    const IntArrayRef& input,
    const IntArrayRef& weight,
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
  // graph_handle scope
  synGraphHandle graph_handle;
  TORCH_HABANA_CHECK(
      synGraphCreate(&graph_handle, synDeviceType::synDeviceGaudi),
      "synGraphCreate failed");
  { // tensors scope
    std::vector<synTensor> syn_inputs(3); // input, filter, bias
    std::vector<synTensor> syn_outputs(1); //  output
    const std::vector<std::string> input_names{"input", "filter", "bias"};
    const std::vector<std::string> output_names{"output"};

    const std::vector<synTensorDescriptorTr> syn_input_descriptors{
        synapse_tensor_descriptor_builder(
            input.sizes(), synDataType::syn_type_float, input_names[0], true),
        synapse_tensor_descriptor_builder(
            weight.sizes(), synDataType::syn_type_float, input_names[1], true),
        synapse_tensor_descriptor_builder(
            bias.sizes(), synDataType::syn_type_float, input_names[2], true)};

    const std::vector<synTensorDescriptorTr> syn_output_descriptors{
        synapse_tensor_descriptor_builder(
            output.sizes(),
            synDataType::syn_type_float,
            output_names[0],
            true)};

    for (int i = 0; i < syn_inputs.size(); ++i)
      TORCH_HABANA_CHECK(
          synTensorCreate(&syn_inputs[i], &syn_input_descriptors[i]),
          "synTensorCreate failed");
    for (int i = 0; i < syn_outputs.size(); ++i)
      TORCH_HABANA_CHECK(
          synTensorCreate(&syn_outputs[i], &syn_output_descriptors[i]),
          "synTensorCreate failed");

    { // dimshuffled tensors scope
#ifdef TRANSPOSE_IMPLEMENTED
      // input, filter Note: I will use original bias
      std::vector<synTensor> syn_tmp_inputs(2);
      std::vector<synTensor> syn_tmp_outputs(1); //  output
      const std::vector<std::string> input_tmp_names{"input_tmp", "filter_tmp"};
      const std::vector<std::string> output_tmp_names{"output_tmp"};

      const std::vector<synTensorDescriptorTr> syn_input_tmp_descriptors{
          synapse_tensor_descriptor_builder(
              NCHW_to_NHWC_shape(input.sizes()),
              synDataType::syn_type_float,
              input_tmp_names[0],
              false),
          synapse_tensor_descriptor_builder(
              {weight.sizes()[2], // KCHW -> HWCK
               weight.sizes()[3],
               weight.sizes()[1],
               weight.sizes()[0]},
              synDataType::syn_type_float,
              input_tmp_names[1],
              false)};
      const std::vector<synTensorDescriptorTr> syn_output_tmp_descriptors{
          synapse_tensor_descriptor_builder(
              NCHW_to_NHWC_shape(output.sizes()),
              synDataType::syn_type_float,
              output_tmp_names[0],
              false)};

      for (int i = 0; i < syn_tmp_inputs.size(); ++i)
        TORCH_HABANA_CHECK(
            synTensorCreate(&syn_tmp_inputs[i], &syn_input_tmp_descriptors[i]),
            "synTensorCreate failed");
      for (int i = 0; i < syn_tmp_outputs.size(); ++i)
        TORCH_HABANA_CHECK(
            synTensorCreate(
                &syn_tmp_outputs[i], &syn_output_tmp_descriptors[i]),
            "synTensorCreate failed");

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
#ifdef TRANSPOSE_IMPLEMENTED
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

      { // graph compilation, workspace buffer and topology buffer allocation
        synRecipeHandle recipe_handle;
        const auto recipe_name = unique_recipe_name_generator(conv_node_type);
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

        auto hpu_RAII_allocator = at::habana::getHABANADeviceAllocator();

        at::DataPtr workspace_buffer,
            topology_buffer = hpu_RAII_allocator->allocate(topology_size_bytes);
        if (workspace_size_bytes)
          workspace_buffer = hpu_RAII_allocator->allocate(workspace_size_bytes);
        { // recipe upload scope
          const synRecipeInfo recipe_info{
              recipe_name.c_str(),
              reinterpret_cast<uint64_t>(topology_buffer.get())};
          const auto device_idx = input.device().index();
          TORCH_HABANA_CHECK(
              synRecipeUpload(recipe_handle, &recipe_info, device_idx),
              "synRecipeUpload failed");
          { // stream handle scope
            synStreamHandle stream_handle;
            TORCH_HABANA_CHECK(
                synStreamCreate(&stream_handle, device_idx, 0),
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
              synRecipeUnload(recipe_handle, &recipe_info, device_idx),
              "synRecipeUnload failed");
        }
      }
#ifdef TRANSPOSE_IMPLEMENTED
      for (int i = 0; i < syn_tmp_inputs.size(); ++i)
        TORCH_HABANA_CHECK(
            synTensorDestroy(syn_tmp_inputs[i]), "synTensorDestroy failed");
      for (int i = 0; i < syn_tmp_outputs.size(); ++i)
        TORCH_HABANA_CHECK(
            synTensorDestroy(syn_tmp_outputs[i]), "synTensorDestroy failed");
#endif
    }

    for (int i = 0; i < syn_inputs.size(); ++i)
      TORCH_HABANA_CHECK(
          synTensorDestroy(syn_inputs[i]), "synTensorDestroy failed");
    for (int i = 0; i < syn_outputs.size(); ++i)
      TORCH_HABANA_CHECK(
          synTensorDestroy(syn_outputs[i]), "synTensorDestroy failed");
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
  std::cout << "habana_convolution called\n";

  { // check dimensions
    TORCH_CHECK(groups == 1, "habana_convolution doesn't support groups");
    TORCH_CHECK(
        transposed == false,
        "habana_convolution doesn't support transposition");
    TORCH_CHECK(
        std::all_of(
            dilation.cbegin(),
            dilation.cend(),
            [](int64_t x) { return x == 1; }),
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
        stride.size() == 2,
        "stride size != 2 unsupported by habana_convolution");
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
    //   at::native::check_shape_forward(input, weight, bias, params, false);
  }
  // input, output NCHW
  // weight KCHW, where K - output channels
  // pad, stride HW
  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  const int64_t input_H = input.size(2);
  const int64_t input_W = input.size(3);
  const int64_t K = weight.size(0);
  TORCH_CHECK(
      weight.size(1) == C,
      "Number of input channels doesn't match weight channels");
  const int64_t filter_H = weight.size(2);
  const int64_t filter_W = weight.size(3);
  const int64_t stride_H = stride[0];
  const int64_t stride_W = stride[1];
  const int64_t pad_H = padding[0];
  const int64_t pad_W = padding[1];
  const auto output_H = compute_output_size(input_H, pad_H, filter_H, stride_H);
  const auto output_W = compute_output_size(input_W, pad_W, filter_W, stride_W);
  std::cout << "input_size N " << N << ", C " << C << ", H " << input_H
            << ", W " << input_W << '\n';
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

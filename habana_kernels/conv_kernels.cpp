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
#include <synapse_api.h>
#include <torch/script.h>
#include <iostream>
#include <string>

#include "conv_pool_utils.h"
#include "habana_device/HPUCheck.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/conv_kernels.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "kernel_utils.h"

using namespace torch;

synConvolutionParams synapse_conv_params_builder(
    const IntArrayRef& weight, // HWCK
    const IntArrayRef& stride, // HW
    const IntArrayRef& padding, // HW
    const IntArrayRef& dilation // HW
) {
  const int64_t filter_H = weight[0];
  const int64_t filter_W = weight[1];
  const int64_t stride_H = stride[0];
  const int64_t stride_W = stride[1];
  const int64_t dilation_H = dilation[0];
  const int64_t dilation_W = dilation[1];

  synConvolutionParams syn_conv_params{};
  syn_conv_params.dH = stride_H;
  syn_conv_params.dW = stride_W;
  syn_conv_params.kH = filter_H;
  syn_conv_params.kW = filter_W;
  syn_conv_params.dilH = dilation_H;
  syn_conv_params.dilW = dilation_W;
  syn_conv_params.setPadT(padding[0]);
  syn_conv_params.setPadB(padding[0]);
  syn_conv_params.setPadL(padding[1]);
  syn_conv_params.setPadR(padding[1]);

  return syn_conv_params;
}

ConvOperator::ConvOperator(int device_id, c10::ScalarType scalarType)
    : HabanaOperator("spatial_convolution") {
  this->CreateSynContext(device_id);
  scalarType_ = scalarType;
  kernel_meta_data_.input_layout.assign({habana::LayoutFormat::NHWC,
                                         habana::LayoutFormat::HWCK,
                                         habana::LayoutFormat::ANY});
  kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NHWC});
}

void ConvOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 9,
      "Incorrect size of inpust expected for Conv operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be int");
  TORCH_CHECK(inputs[3].isIntList(), "Input type expected to be IntList");
  TORCH_CHECK(inputs[4].isIntList(), "Input type expected to be IntList");
  TORCH_CHECK(inputs[5].isIntList(), "Input type expected to be IntList");
  TORCH_CHECK(inputs[6].isBool(), "Input type expected to be bool");
  TORCH_CHECK(inputs[7].isIntList(), "Input type expected to be IntList");
  TORCH_CHECK(inputs[8].isInt(), "Input type expected to be Int");

  at::Tensor bias;
  at::Tensor input = inputs[0].toTensor();
  at::Tensor weight = inputs[1].toTensor();
  const auto stride = inputs[3].toIntList().vec();
  const auto padding = inputs[4].toIntList().vec();
  const auto dilation = inputs[5].toIntList().vec();
  const bool transposed = inputs[6].toBool();
  const auto output_padding = inputs[7].toIntList().vec();
  const int64_t groups = inputs[8].toInt();

  // bias input is optional, it can either be a Tensor or should be None
  if (inputs[2].isTensor()) {
    bias = inputs[2].toTensor();
  } else {
    TORCH_CHECK(
        inputs[2].isNone(), "Input[2]/bias is either None or Tensor Type");
  }

  std::vector<at::Tensor> pt_inputs{input, weight};
  if (bias.defined()) {
    pt_inputs.emplace_back(bias);
  }
  habana_helpers::check_convolution_params(
      pt_inputs,
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(dilation),
      transposed,
      IntArrayRef(output_padding),
      groups,
      3 /*input_channel*/,
      2 /*weight_channel*/);

  // input, output NCHW
  // weight KCHW, where K - output channels
  // pad, stride HW
  const int64_t N = input.size(0);
  const int64_t input_H = input.size(1);
  const int64_t input_W = input.size(2);
  const int64_t K = weight.size(3);
  const int64_t filter_H = weight.size(0);
  const int64_t filter_W = weight.size(1);
  const int64_t stride_H = stride[0];
  const int64_t stride_W = stride[1];
  const int64_t pad_H = padding[0];
  const int64_t pad_W = padding[1];
  const auto output_H = habana_helpers::compute_output_size(
      input_H, pad_H, filter_H, stride_H, false);
  const auto output_W = habana_helpers::compute_output_size(
      input_W, pad_W, filter_W, stride_W, false);

  c10::MemoryFormat memory_format =
      habana_helpers::get_memory_format({&input, &weight});

  auto output =
      at::empty({N, output_H, output_W, K}, input.options(), memory_format);

  synConvolutionParams params = synapse_conv_params_builder(
      weight.sizes(),
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(dilation));

  p_context_->params_.emplace<synConvolutionParams>(params);
  p_context_->params_size_ = sizeof(params);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

Tensor convolution_hpu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups) {
  LOG_FUNC_BEGIN;
  std::vector<at::Tensor> inputs{input, weight};
  if (bias.defined()) {
    inputs.push_back(bias);
  }
  // convert tensors to synapse memory format
  Tensor input_nhwc = input;
  Tensor weight_hwck = weight;
  std::vector<const at::Tensor*> pt_in{&input, &weight};
  std::vector<at::Tensor*> pt_out{&input_nhwc, &weight_hwck};
  IntArrayRef new_dim_pos_in = {0, 2, 3, 1};
  IntArrayRef new_dim_pos_w = {2, 3, 1, 0};
  std::vector<const IntArrayRef*> pt_new_pos{&new_dim_pos_in, &new_dim_pos_w};
  c10::MemoryFormat memory_format =
      habana_helpers::get_memory_format({&input, &weight});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  auto convolution = [&] {
    size_t device_id = input.device().index();
    at::ScalarType scalar_type = input.scalar_type();
    std::string node_type = "spatial_convolution";

    //
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Create the operator
    ConvOperator Op(device_id, scalar_type);

    // Assign Inputs to the Operator
    std::vector<const at::Tensor*> pt_inputs{&input_nhwc, &weight_hwck};
    if (bias.defined()) {
      pt_inputs.emplace_back(&bias);
    }
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    std::vector<c10::IValue> stack = {IValue(input_nhwc),
                                      IValue(weight_hwck),
                                      IValue(bias),
                                      IValue(stride),
                                      IValue(padding),
                                      IValue(dilation),
                                      IValue(transposed),
                                      IValue(output_padding),
                                      IValue(groups)};
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    Op.Compile(graph);

    std::vector<at::Tensor> out = Op.GetOutputs();
    TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
    return out[0];
  };

  Tensor output;
  auto output_nhwc = convolution();
  pt_in = {&output_nhwc};
  pt_out = {&output};
  IntArrayRef new_dim_pos_out = {0, 3, 1, 2};
  pt_new_pos = {&new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  LOG_FUNC_END;
  return output;
}

Tensor convolution_backward_input(
    const IntArrayRef input_sizes_nhwc,
    const Tensor& weight_hwck,
    const Tensor& grad_output_nhwc,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    c10::MemoryFormat memory_format) {
  auto grad_input_nhwc =
      at::empty(input_sizes_nhwc, grad_output_nhwc.options(), memory_format);

  synConvolutionParams syn_params = synapse_conv_params_builder(
      weight_hwck.sizes(), stride, padding, dilation);

  std::vector<const at::Tensor*> pt_outputs{&grad_input_nhwc};
  std::vector<const at::Tensor*> pt_inputs{&grad_output_nhwc, &weight_hwck};
  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "dedx",
      &syn_params,
      sizeof(syn_params),
      SynapsePassType::NO_PASS);

  Tensor grad_in = grad_input_nhwc;
  std::vector<const at::Tensor*> pt_in = {&grad_input_nhwc};
  std::vector<at::Tensor*> pt_out = {&grad_in};
  IntArrayRef new_dim_pos_out = {0, 3, 1, 2};
  std::vector<const IntArrayRef*> pt_new_pos = {&new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  return grad_in;
}

Tensor convolution_backward_filter(
    const Tensor& input_nhwc,
    const IntArrayRef weight_size_hwck,
    const Tensor& grad_out_nhwc,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    c10::MemoryFormat memory_format) {
  auto grad_weight =
      at::empty(weight_size_hwck, grad_out_nhwc.options(), memory_format);
  synConvolutionParams syn_params = synapse_conv_params_builder(
      grad_weight.sizes(), stride, padding, dilation);

  std::vector<const at::Tensor*> pt_outputs{&grad_weight};
  std::vector<const at::Tensor*> pt_inputs{&grad_out_nhwc, &input_nhwc};
  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "dedw",
      &syn_params,
      sizeof(syn_params),
      SynapsePassType::NO_PASS);

  Tensor grad_w = grad_weight;
  std::vector<const at::Tensor*> pt_in = {&grad_weight};
  std::vector<at::Tensor*> pt_out = {&grad_w};
  IntArrayRef new_dim_pos_out = {3, 2, 0, 1};
  std::vector<const IntArrayRef*> pt_new_pos = {&new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  return grad_w;
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward_hpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  LOG_FUNC_BEGIN;
  std::vector<at::Tensor> inputs{input, weight};
  habana_helpers::check_convolution_params(
      inputs, stride, padding, dilation, transposed, output_padding, groups);

  // pad, stride HW
  const int64_t input_H = input.size(2);
  const int64_t input_W = input.size(3);
  const int64_t filter_H = weight.size(2);
  const int64_t filter_W = weight.size(3);
  const int64_t stride_H = stride[0];
  const int64_t stride_W = stride[1];
  const int64_t pad_H = padding[0];
  const int64_t pad_W = padding[1];
  const int64_t output_H = grad_output.size(2);
  const int64_t output_W = grad_output.size(3);
  TORCH_CHECK(
      output_H ==
      habana_helpers::compute_output_size(
          input_H, pad_H, filter_H, stride_H, false));
  TORCH_CHECK(
      output_W ==
      habana_helpers::compute_output_size(
          input_W, pad_W, filter_W, stride_W, false));

  // convert tensors to synapse memory format
  Tensor input_nhwc = input;
  Tensor grad_out_nhwc = grad_output;
  Tensor weight_hwck = weight;
  std::vector<const at::Tensor*> pt_in{&input, &grad_output, &weight};
  std::vector<at::Tensor*> pt_out{&input_nhwc, &grad_out_nhwc, &weight_hwck};
  IntArrayRef new_dim_pos_in = {0, 2, 3, 1};
  IntArrayRef new_dim_pos_grad_out = {0, 2, 3, 1};
  IntArrayRef new_dim_pos_w = {2, 3, 1, 0};
  std::vector<const IntArrayRef*> pt_new_pos{
      &new_dim_pos_in, &new_dim_pos_grad_out, &new_dim_pos_w};
  c10::MemoryFormat memory_format =
      habana_helpers::get_memory_format({&grad_output, &input, &weight});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  Tensor grad_input, grad_weight, grad_bias;

  if (output_mask[0])
    grad_input = convolution_backward_input(
        input_nhwc.sizes(),
        weight_hwck,
        grad_out_nhwc,
        stride,
        padding,
        dilation,
        memory_format);
  if (output_mask[1]) {
    grad_weight = convolution_backward_filter(
        input_nhwc,
        weight_hwck.sizes(),
        grad_out_nhwc,
        stride,
        padding,
        dilation,
        memory_format);
  }
  if (output_mask[2]) {
    grad_bias = grad_out_nhwc;
    std::vector<int64_t> dim_to_reduce;
    for (int64_t i = 0; i < grad_out_nhwc.ndimension(); ++i) {
      if (i != 3) // skip C dimension
        dim_to_reduce.push_back(i);
    }
    c10::IntArrayRef shape(dim_to_reduce.data(), dim_to_reduce.size());
    grad_bias = grad_bias.sum(shape, false);

    TORCH_CHECK(
        grad_bias.numel() == grad_out_nhwc.size(3),
        "Bias grad numelements must equal to number of conv output channels. Got: ",
        grad_bias.numel(),
        "expected: ",
        grad_out_nhwc.size(3));
  }
  LOG_FUNC_END;
  return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(convolution_hpu),
                    &convolution_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")
                .impl_unboxedOnlyKernel<
                    decltype(convolution_backward_hpu),
                    &convolution_backward_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
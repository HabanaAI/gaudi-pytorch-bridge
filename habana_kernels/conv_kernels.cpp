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
#include "habana_kernels/reduction_kernels.h"
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
  kernel_meta_data_.input_layout.assign(
      {habana::LayoutFormat::NHWC,
       habana::LayoutFormat::HWCK,
       habana::LayoutFormat::ANY});
  kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NHWC});
}

std::vector<int64_t> ConvOperator::compute_output_shape(
    std::vector<int64_t> shape_in,
    std::vector<int64_t> shape_wt,
    std::vector<int64_t> pad,
    std::vector<int64_t> stride,
    const bool ceil_mode,
    c10::MemoryFormat memory_format) {
  HABANA_ASSERT(ceil_mode == false);
  HABANA_ASSERT(
      (memory_format == c10::MemoryFormat::ChannelsLast) ||
      (memory_format == c10::MemoryFormat::Contiguous));

  const int64_t dim_pos_in[4] = {0, 1, 2, 3};
  const int64_t dim_pos_wt[4] = {0, 1, 2, 3};
  const int64_t dim_pos_in_chlast[4] = {0, 2, 3, 1};
  const int64_t dim_pos_wt_chlast[4] = {2, 3, 1, 0};
  const int64_t* p_dim_pos_in;
  const int64_t* p_dim_pos_wt;

  if (memory_format == c10::MemoryFormat::ChannelsLast) {
    p_dim_pos_in = dim_pos_in_chlast;
    p_dim_pos_wt = dim_pos_wt_chlast;
  } else {
    p_dim_pos_in = dim_pos_in;
    p_dim_pos_wt = dim_pos_wt;
  }

  const auto input_H = shape_in[p_dim_pos_in[1]];
  const auto pad_H = pad[0];
  const auto filter_H = shape_wt[p_dim_pos_wt[0]];
  const auto stride_H = stride[0];

  const auto output_H = habana_helpers::compute_output_size(
      input_H, pad_H, filter_H, stride_H, false);

  const auto input_W = shape_in[p_dim_pos_in[2]];
  const auto pad_W = pad[1];
  const auto filter_W = shape_wt[p_dim_pos_wt[1]];
  const auto stride_W = stride[1];
  const auto output_W = habana_helpers::compute_output_size(
      input_W, pad_W, filter_W, stride_W, false);

  const auto K = shape_wt[p_dim_pos_wt[3]];

  std::vector<int64_t> out_shape = {shape_in[0], output_H, output_W, K};
  return out_shape;
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
  c10::MemoryFormat memory_format =
      habana_helpers::get_memory_format({&input, &weight});

  std::vector<int64_t> shape_out = compute_output_shape(
      input.sizes().vec(),
      weight.sizes().vec(),
      padding,
      stride,
      false,
      memory_format);

  auto output = habana_helpers::createPTTensor(
      input, shape_out, input.options(), memory_format, is_output_persistent);

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

void ConvOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  at::Tensor input = inputs[0].toTensor();
  at::Tensor weight = inputs[1].toTensor();
  const auto stride = inputs[3].toIntList().vec();
  const auto padding = inputs[4].toIntList().vec();

  c10::MemoryFormat memory_format =
      habana_helpers::get_memory_format({&input, &weight});

  auto shape_out = compute_output_shape(
      input.sizes().vec(),
      weight.sizes().vec(),
      padding,
      stride,
      false,
      memory_format);

  auto output =
      at::empty(shape_out, input.options(), memory_format);
  HabanaOperator::SetPTOutputs({output});
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
  PT_KERNEL_BEGIN;
  std::vector<at::Tensor> inputs{input, weight};
  if (bias.defined()) {
    inputs.push_back(bias);
  }
  // convert tensors to synapse memory format
  Tensor input_nhwc = input;
  Tensor weight_hwck = weight;
  std::vector<const at::Tensor*> pt_in{&input};
  std::vector<at::Tensor*> pt_out{&input_nhwc};
  IntArrayRef new_dim_pos_in = {0, 2, 3, 1};
  IntArrayRef new_dim_pos_w = {2, 3, 1, 0};
  std::vector<const IntArrayRef*> pt_new_pos{&new_dim_pos_in, &new_dim_pos_w};
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  // habana_helpers::change_tensor_strides(&weight_hwck, &weight,
  // &new_dim_pos_w);

  auto convolution = [&] {
    size_t device_id = input.device().index();
    at::ScalarType scalar_type = input.scalar_type();
    std::string node_type = "spatial_convolution";
    auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
    // Create the operator
    ConvOperator Op(device_id, scalar_type);

    // Build Params for the graph
    std::vector<c10::IValue> stack = {
        IValue(input_nhwc),
        IValue(weight_hwck),
        IValue(bias),
        IValue(stride),
        IValue(padding),
        IValue(dilation),
        IValue(transposed),
        IValue(output_padding),
        IValue(groups)};
    size_t key = Op.GetRecipeKey(node_type, stack);

    // Assign Inputs to the Operator
    std::vector<at::Tensor> pt_inputs{input_nhwc, weight_hwck};
    if (bias.defined()) {
      pt_inputs.emplace_back(bias);
    }

    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      Op.SetPTInputs(pt_inputs);
      Op.SetPTOutputs(stack);
      Op.Execute(key);
    } else {
      PT_KERNEL_DEBUG("key:", key);
      //
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);
      Op.AllocateSynapseInputs(graph, pt_inputs, true);
      Op.AllocateAndAddSynapseNode(graph, stack, true);
      Op.Compile(graph);
    }

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
  PT_KERNEL_END;
  return output;
}

void ConvInputDifferentiationOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 9,
      "Incorrect size of inputs expected for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be tensor for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[3].isIntList(),
      "Input arg4 expected to be IntList for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[4].isIntList(),
      "Input arg5 expected to be IntList for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[5].isIntList(),
      "Input arg6 expected to be IntList for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[6].isIntList(),
      "Input arg7 expected to be IntList for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[7].isBoolList(),
      "Input arg8 expected to be BoolList for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[8].isTensor(),
      "Input arg9 expected to be tensor for ConvInputDifferentiation operator");

  auto grad_out_nhwc = inputs[0].toTensor();
  auto input_nhwc = inputs[1].toTensor();
  auto weight_hwck = inputs[2].toTensor();
  const auto stride = inputs[3].toIntList().vec();
  const auto padding = inputs[4].toIntList().vec();
  const auto dilation = inputs[5].toIntList().vec();
  auto output_padding = inputs[6].toIntList();
  auto output_mask_in = inputs[7].toBoolList();
  auto grad_input_nhwc = inputs[8].toTensor();

  synConvolutionParams syn_params = synapse_conv_params_builder(
      weight_hwck.sizes(),
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(dilation));

  p_context_->params_.emplace<synConvolutionParams>(syn_params);
  p_context_->params_size_ = sizeof(syn_params);

  AllocateSynapseOutput(graph, grad_input_nhwc, is_output_persistent);
  AddNodeToSynapseGraph(graph, &syn_params, sizeof(syn_params));
}

void ConvWeightDifferentiationOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 9,
      "Incorrect size of inputs expected for ConvWeightDifferentiation operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for ConvWeightDifferentiation operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for ConvWeightDifferentiation operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be tensor for ConvWeightDifferentiation operator");
  TORCH_CHECK(
      inputs[3].isIntList(),
      "Input arg4 expected to be IntList for ConvWeightDifferentiation operator");
  TORCH_CHECK(
      inputs[4].isIntList(),
      "Input arg5 expected to be IntList for ConvWeightDifferentiation operator");
  TORCH_CHECK(
      inputs[5].isIntList(),
      "Input arg6 expected to be IntList for ConvWeightDifferentiation operator");
  TORCH_CHECK(
      inputs[6].isIntList(),
      "Input arg7 expected to be IntList for ConvWeightDifferentiation operator");
  TORCH_CHECK(
      inputs[7].isBoolList(),
      "Input arg8 expected to be BoolList for ConvWeightDifferentiation operator");
  TORCH_CHECK(
      inputs[8].isTensor(),
      "Input arg9 expected to be tensor for ConvInputDifferentiation operator");

  auto grad_out_nhwc = inputs[0].toTensor();
  auto input_nhwc = inputs[1].toTensor();
  auto weight_hwck = inputs[2].toTensor();
  const auto stride = inputs[3].toIntList().vec();
  const auto padding = inputs[4].toIntList().vec();
  const auto dilation = inputs[5].toIntList().vec();
  auto output_padding = inputs[6].toIntList();
  auto output_mask_in = inputs[7].toBoolList();
  auto grad_weight = inputs[8].toTensor();

  synConvolutionParams syn_params = synapse_conv_params_builder(
      grad_weight.sizes(),
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(dilation));

  p_context_->params_.emplace<synConvolutionParams>(syn_params);
  p_context_->params_size_ = sizeof(syn_params);

  AllocateSynapseOutput(graph, grad_weight, is_output_persistent);
  AddNodeToSynapseGraph(graph, &syn_params, sizeof(syn_params));
}

void ConvBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 10,
      "Incorrect size of inputs expected for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be tensor for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[3].isIntList(),
      "Input arg4 expected to be IntList for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[4].isIntList(),
      "Input arg5 expected to be IntList for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[5].isIntList(),
      "Input arg6 expected to be IntList for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[6].isBool(),
      "Input arg7 expected to be Bool for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[7].isIntList(),
      "Input arg8 expected to be IntList for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[8].isInt(),
      "Input arg9 expected to be Int for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[9].isBoolList(),
      "Input arg10 expected to be BoolList for ConvInputDifferentiation operator");
  TORCH_CHECK(
      is_output_persistent.size() == 3,
      "ConvBackwardOperator: #is_output_persistent should be 3");

  auto grad_out_nhwc = inputs[0].toTensor();
  auto input_nhwc = inputs[1].toTensor();
  auto weight_hwck = inputs[2].toTensor();
  const auto stride = inputs[3].toIntList().vec();
  const auto padding = inputs[4].toIntList().vec();
  const auto dilation = inputs[5].toIntList().vec();
  UNUSED auto transposed = inputs[6].toBool();
  auto output_padding = inputs[7].toIntList().vec();
  UNUSED auto groups = inputs[8].toInt();
  auto output_mask_in = inputs[9].toBoolList();

  c10::MemoryFormat memory_format = habana_helpers::get_memory_format(
      {&input_nhwc, &grad_out_nhwc, &weight_hwck});

  auto grad_weight = habana_helpers::createPTTensor(
      weight_hwck,
      weight_hwck.sizes(),
      grad_out_nhwc.options(),
      memory_format,
      is_output_persistent[1]);
  auto grad_input_nhwc = habana_helpers::createPTTensor(
      input_nhwc,
      input_nhwc.sizes(),
      grad_out_nhwc.options(),
      memory_format,
      is_output_persistent[0]);
  auto grad_bias = habana_helpers::createPTTensor(
      grad_out_nhwc,
      {grad_out_nhwc.size(3)},
      grad_out_nhwc.options(),
      c10::nullopt,
      is_output_persistent[2]);

  // Add "dedw" node followed by "dedx" node. Adding in reverse order causes a
  // simulator crash (TBD: investigate later if required)

  // Create the operator
  std::string node_type = "dedw";
  ConvWeightDifferentiationOperator ConvWeightDiffOp(
      this->p_context_->device_id_, node_type);
  if (output_mask_in[1]) {
    // Assign Inputs to the Operator
    auto& grad_out_nhwc_syn =
        ConvWeightDiffOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    auto& input_nhwc_syn =
        ConvWeightDiffOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));

    // Build Params for the graph
    std::vector<c10::IValue> stack = {
        IValue(grad_out_nhwc),
        IValue(input_nhwc),
        IValue(weight_hwck),
        IValue(stride),
        IValue(padding),
        IValue(dilation),
        IValue(output_padding),
        IValue(output_mask_in),
        IValue(grad_weight)};
    ConvWeightDiffOp.AllocateAndAddSynapseNode(
        graph, stack, is_output_persistent[1]);

    p_context_->syn_inputs_[0] = std::move(grad_out_nhwc_syn);
    p_context_->syn_inputs_[1] = std::move(input_nhwc_syn);
  }

  // Create the operator
  node_type = "dedx";
  ConvInputDifferentiationOperator ConvInputDiffOp(
      this->p_context_->device_id_, node_type);
  if (output_mask_in[0]) {
    // Assign Inputs to the Operator
    auto& grad_out_nhwc_syn =
        ConvInputDiffOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    auto& weight_hwck_syn =
        ConvInputDiffOp.SetSynapseInput(std::move(p_context_->syn_inputs_[2]));

    // Build Params for the graph
    std::vector<c10::IValue> stack = {
        IValue(grad_out_nhwc),
        IValue(input_nhwc),
        IValue(weight_hwck),
        IValue(stride),
        IValue(padding),
        IValue(dilation),
        IValue(output_padding),
        IValue(output_mask_in),
        IValue(grad_input_nhwc)};
    ConvInputDiffOp.AllocateAndAddSynapseNode(
        graph, stack, is_output_persistent[0]);

    p_context_->syn_inputs_[0] = std::move(grad_out_nhwc_syn);
    p_context_->syn_inputs_[2] = std::move(weight_hwck_syn);
  }

  // Although we have "dedw" node first in the graph followed by "dedw", when
  // pushing outputs we want to maintain correct order
  if (output_mask_in[0]) {
    synapse_helpers::tensor& grad_in_nhwc_syn_tensor =
        ConvInputDiffOp.GetSynOutputs()[0];
    p_context_->syn_outputs_.emplace_back(std::move(grad_in_nhwc_syn_tensor));
    p_context_->pt_outputs_.emplace_back(
        std::move(ConvInputDiffOp.GetOutputs()[0]));
  } else {
    p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
        grad_input_nhwc,
        graph.get_graph_handle(),
        is_output_persistent[0],
        c10::nullopt));
    p_context_->pt_outputs_.emplace_back(grad_input_nhwc);
  }

  if (output_mask_in[1]) {
    synapse_helpers::tensor& grad_weight_syn_tensor =
        ConvWeightDiffOp.GetSynOutputs()[0];

    p_context_->syn_outputs_.emplace_back(std::move(grad_weight_syn_tensor));
    p_context_->pt_outputs_.emplace_back(
        std::move(ConvWeightDiffOp.GetOutputs()[0]));
  } else {
    p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
        grad_weight,
        graph.get_graph_handle(),
        is_output_persistent[1],
        c10::nullopt));
    p_context_->pt_outputs_.emplace_back(grad_weight);
  }

  if (output_mask_in[2]) {
    std::vector<int64_t> dim_to_reduce;
    for (int64_t i = 0; i < grad_out_nhwc.ndimension(); ++i) {
      if (i != 3) // skip C dimension
        dim_to_reduce.push_back(i);
    }
    c10::IntArrayRef shape(dim_to_reduce.data(), dim_to_reduce.size());

    at::ScalarType scalar_type = grad_out_nhwc.scalar_type();
    std::string node_type =
        "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

    // Create the operator
    SumDimOutOperator SumOp(this->p_context_->device_id_, node_type);

    // Assign Inputs to the Operator
    auto& grad_out_nhwc_syn =
        SumOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));

    std::vector<c10::IValue> stack = {
        IValue(grad_bias),
        IValue(grad_out_nhwc),
        IValue(shape),
        IValue(false),
        IValue(scalar_type)};
    SumOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent[0]);

    synapse_helpers::tensor& bias_syn_tensor = SumOp.GetSynOutputs()[0];

    p_context_->syn_inputs_[0] = std::move(grad_out_nhwc_syn);

    p_context_->syn_outputs_.emplace_back(std::move(bias_syn_tensor));
    p_context_->pt_outputs_.emplace_back(std::move(SumOp.GetOutputs()[0]));

  } else {
    p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
        grad_bias,
        graph.get_graph_handle(),
        is_output_persistent[2],
        c10::nullopt));
    p_context_->pt_outputs_.emplace_back(grad_bias);
  }
}

void ConvBackwardOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto grad_out_nhwc = inputs[0].toTensor();
  auto input_nhwc = inputs[1].toTensor();
  auto weight_hwck = inputs[2].toTensor();
  const auto stride = inputs[3].toIntList().vec();
  const auto padding = inputs[4].toIntList().vec();
  const auto dilation = inputs[5].toIntList().vec();
  UNUSED auto transposed = inputs[6].toBool();
  auto output_padding = inputs[7].toIntList();
  UNUSED auto groups = inputs[8].toInt();
  auto output_mask_in = inputs[9].toBoolList();

  c10::MemoryFormat memory_format = habana_helpers::get_memory_format(
      {&input_nhwc, &grad_out_nhwc, &weight_hwck});

  auto grad_weight =
      at::empty(weight_hwck.sizes(), grad_out_nhwc.options(), memory_format);
  auto grad_input_nhwc =
      at::empty(input_nhwc.sizes(), grad_out_nhwc.options(), memory_format);
  auto grad_bias = at::empty(
      {grad_out_nhwc.size(3)}, grad_out_nhwc.options(), memory_format);

  HabanaOperator::SetPTOutput(grad_input_nhwc);
  HabanaOperator::SetPTOutput(grad_weight);
  HabanaOperator::SetPTOutput(grad_bias);
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
  PT_KERNEL_BEGIN;

  std::vector<bool> output_mask_in;
  output_mask_in.push_back(output_mask[0]);
  output_mask_in.push_back(output_mask[1]);
  output_mask_in.push_back(output_mask[2]);

  std::vector<at::Tensor> inputs{input, weight};
  unsigned int input_channel_index = 1;
  habana_helpers::check_convolution_params(
      inputs,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      input_channel_index,
      2);

  // pad, stride HW
  const int64_t input_H = input.size(2);
  const int64_t input_W = input.size(3);
  const int64_t filter_H = weight.size(0);
  const int64_t filter_W = weight.size(1);
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
  std::vector<const at::Tensor*> pt_in{&input, &grad_output};
  std::vector<at::Tensor*> pt_out{&input_nhwc, &grad_out_nhwc};
  IntArrayRef new_dim_pos_in = {0, 2, 3, 1};
  IntArrayRef new_dim_pos_grad_out = {0, 2, 3, 1};
  IntArrayRef new_dim_pos_w = {2, 3, 1, 0};
  std::vector<const IntArrayRef*> pt_new_pos{
      &new_dim_pos_in, &new_dim_pos_grad_out, &new_dim_pos_w};
  c10::MemoryFormat memory_format =
      habana_helpers::get_memory_format({&grad_output, &input});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  // habana_helpers::change_tensor_strides(&weight_hwck, &weight,
  // &new_dim_pos_w);

  Tensor grad_input, grad_weight, grad_bias;

  auto convolution_backward = [&] {
    std::string node_type = "convolution_bwd";

    // Create the operator
    size_t device_id = grad_out_nhwc.device().index();
    auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

    ConvBackwardOperator convBwdOp(device_id, input_nhwc.scalar_type());

    // Build Params for the graph
    std::vector<c10::IValue> stack = {
        IValue(grad_out_nhwc),
        IValue(input_nhwc),
        IValue(weight_hwck),
        IValue(stride),
        IValue(padding),
        IValue(dilation),
        IValue(transposed),
        IValue(output_padding),
        IValue(groups),
        IValue(output_mask_in),
    };
    size_t key = convBwdOp.GetRecipeKey(node_type, stack);

    // Assign Inputs to the Operator
    std::vector<at::Tensor> pt_inputs{grad_out_nhwc, input_nhwc, weight_hwck};

    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      convBwdOp.SetPTInputs(pt_inputs);
      convBwdOp.SetPTOutputs(stack);
      convBwdOp.Execute(key);
    } else {
      PT_KERNEL_DEBUG("key:", key);
      //
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);
      convBwdOp.AllocateSynapseInputs(graph, pt_inputs, true);
      convBwdOp.AllocateAndAddSynapseNode(graph, stack, {true, true, true});
      convBwdOp.Compile(graph);
    }

    std::vector<at::Tensor> output = convBwdOp.GetOutputs();
    return output;
  };

  std::vector<at::Tensor> conv_out = convolution_backward();

  auto grad_input_nhwc = conv_out.at(0);
  auto grad_weight_hwck = conv_out.at(1);
  grad_bias = conv_out.at(2);

  if (output_mask[0]) {
    Tensor grad_in;
    pt_in = {&grad_input_nhwc};
    pt_out = {&grad_in};
    IntArrayRef new_dim_pos_out = {0, 3, 1, 2};
    std::vector<const IntArrayRef*> pt_new_pos = {&new_dim_pos_out};
    habana_helpers::change_tensors_to_memory_format(
        pt_out, pt_in, pt_new_pos, memory_format);
    grad_input = grad_in;
  }

  if (output_mask[1]) {
    grad_weight = grad_weight_hwck;
  }

  if (output_mask[2]) {
    TORCH_CHECK(
        grad_bias.numel() == grad_out_nhwc.size(3),
        "Bias grad numelements must equal to number of conv output channels. Got: ",
        grad_bias.numel(),
        "expected: ",
        grad_out_nhwc.size(3));
  }
  PT_KERNEL_END;

  return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::convolution_overrideable",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ConvOperator>(device_id, node_type);
            })
        .add(
            "aten::conv2d",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<Conv2dOperator>(device_id, node_type);
            })
        .add(
            "aten::convolution_backward_overrideable",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ConvBackwardOperator>(
                  device_id, node_type);
            });

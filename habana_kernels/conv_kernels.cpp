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
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/conv_kernels.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "kernel_utils.h"

using namespace torch;
using namespace habana;

synConvolutionParams synapse_conv_params_builder(
    const IntArrayRef& weight, // HWCK
    const IntArrayRef& stride, // HW
    const IntArrayRef& padding, // HW
    const IntArrayRef& dilation, // HW
    int64_t groups) {
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
  syn_conv_params.nGroups = groups;

  return syn_conv_params;
}

/**
 * @brief computes output shape for conv kernels
 * @param shape_in <in> - NCHW if memory_format = contiguous. NHWC for
 *ChannelsLast
 * @param shape_wt <in> - HWCK (conv2d), HWKC (conv_transpose2d)
 * @param pad <in> - HW
 * @param stride <in> - HW
 * @param ceil_mode <in>
 * @param transposed <in>
 * @param out_memory_format - Output mem format (contigous - NCHW, ChannelsLast
 *- NHWC)
 * @param out_shape <out> - NCHW/NHWC depending on output mem format
 **/
std::vector<int64_t> ConvOperator::compute_output_shape(
    std::vector<int64_t> shape_in,
    std::vector<int64_t> shape_wt,
    std::vector<int64_t> pad,
    std::vector<int64_t> stride,
    const bool ceil_mode,
    const bool transposed,
    c10::MemoryFormat memory_format) {
  HABANA_ASSERT(ceil_mode == false);
  HABANA_ASSERT(
      (memory_format == c10::MemoryFormat::ChannelsLast) ||
      (memory_format == c10::MemoryFormat::Contiguous));

  const int64_t dim_pos_in[4] = {0, 2, 3, 1};
  const int64_t dim_pos_wt[4] = {0, 1, 2, 3};
  const int64_t dim_pos_in_chlast[4] = {0, 1, 2, 3};
  const int64_t* p_dim_pos_in;
  const int64_t* p_dim_pos_wt;

  if (memory_format == c10::MemoryFormat::ChannelsLast) {
    p_dim_pos_in = dim_pos_in_chlast;
  } else {
    p_dim_pos_in = dim_pos_in;
  }

  p_dim_pos_wt = dim_pos_wt;

  const auto input_H = shape_in[p_dim_pos_in[1]];
  const auto pad_H = pad[0];
  const auto filter_H = shape_wt[p_dim_pos_wt[0]];
  const auto stride_H = stride[0];

  const auto output_H = habana_helpers::compute_output_size(
      input_H, pad_H, filter_H, stride_H, false, transposed);

  const auto input_W = shape_in[p_dim_pos_in[2]];
  const auto pad_W = pad[1];
  const auto filter_W = shape_wt[p_dim_pos_wt[1]];
  const auto stride_W = stride[1];
  const auto output_W = habana_helpers::compute_output_size(
      input_W, pad_W, filter_W, stride_W, false, transposed);

  auto K = shape_wt[p_dim_pos_wt[3]];
  if (transposed) {
    // for conv_transpose2d weights are in HWKC format
    K = shape_wt[p_dim_pos_wt[2]];
  }
  std::vector<int64_t> out_shape;

  if (memory_format == c10::MemoryFormat::ChannelsLast) {
    out_shape.push_back(shape_in[0]);
    out_shape.push_back(output_H);
    out_shape.push_back(output_W);
    out_shape.push_back(K);
  } else {
    out_shape.push_back(shape_in[0]);
    out_shape.push_back(K);
    out_shape.push_back(output_H);
    out_shape.push_back(output_W);
  }
  return out_shape;
}

void SpatialConvOperator::AllocateAndAddSynapseNode(
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
  if (bias.defined() && !transposed) {
    pt_inputs.emplace_back(bias);
  }

  auto weight_channel = 2;
  if (transposed) {
    // conv_transpose2d realized using conv_backward w.r.t input, so use guid
    // corresponding to that
    SetGuid("dedx");
    // conv_transpose2d weights are in HWKC format
    weight_channel = 3;
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
      weight_channel);

  // input, output NCHW
  // weight KCHW, where K - output channels
  // pad, stride HW
  c10::MemoryFormat memory_format =
      habana_helpers::get_memory_format({&input, &weight});

  // permute happened outside this function. Hence always set channelsLast
  // format
  std::vector<int64_t> shape_out = ConvOperator::compute_output_shape(
      input.sizes().vec(),
      weight.sizes().vec(),
      padding,
      stride,
      false,
      transposed,
      c10::MemoryFormat::ChannelsLast);

  auto output = habana_helpers::createPTTensor(
      input, shape_out, input.options(), memory_format, is_output_persistent);

  synConvolutionParams params = synapse_conv_params_builder(
      weight.sizes(),
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(dilation),
      groups);

  p_context_->params_.emplace<synConvolutionParams>(params);
  p_context_->params_size_ = sizeof(params);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
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

  at::Tensor bias = Tensor();
  at::Tensor input = inputs[0].toTensor();
  // at::Tensor weight = inputs[1].toTensor();
  // const auto stride = inputs[3].toIntList().vec();
  // const auto padding = inputs[4].toIntList().vec();
  // const auto dilation = inputs[5].toIntList().vec();
  const bool transposed = inputs[6].toBool();
  // const auto output_padding = inputs[7].toIntList().vec();
  // const int64_t groups = inputs[8].toInt();

  // bias input is optional, it can either be a Tensor or should be None
  if (inputs[2].isTensor()) {
    bias = inputs[2].toTensor();
  } else {
    TORCH_CHECK(
        inputs[2].isNone(), "Input[2]/bias is either None or Tensor Type");
  }

  if (!bias.defined()) {
    SpatialConvOperator scOp(this->p_context_->device_id_, input.scalar_type());
    auto& syn_arg0 =
        scOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    auto& syn_arg1 =
        scOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
    scOp.AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
    p_context_->syn_inputs_[0] = std::move(syn_arg0);
    p_context_->syn_inputs_[1] = std::move(syn_arg1);

    p_context_->syn_outputs_.emplace_back(std::move(scOp.GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(scOp.GetOutputs()[0]));
  } else {
    if (!transposed) {
      SpatialConvOperator scOp(
          this->p_context_->device_id_, input.scalar_type());
      auto& syn_arg0 =
          scOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
      auto& syn_arg1 =
          scOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
      auto& syn_arg2 =
          scOp.SetSynapseInput(std::move(p_context_->syn_inputs_[2]));
      scOp.AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
      p_context_->syn_inputs_[0] = std::move(syn_arg0);
      p_context_->syn_inputs_[1] = std::move(syn_arg1);
      p_context_->syn_inputs_[2] = std::move(syn_arg2);

      p_context_->syn_outputs_.emplace_back(std::move(scOp.GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(std::move(scOp.GetOutputs()[0]));
    } else {
      SpatialConvOperator scOp(
          this->p_context_->device_id_, input.scalar_type());
      auto& syn_arg0 =
          scOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
      auto& syn_arg1 =
          scOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
      scOp.AllocateAndAddSynapseNode(graph, inputs, false);
      p_context_->syn_inputs_[0] = std::move(syn_arg0);
      p_context_->syn_inputs_[1] = std::move(syn_arg1);

      AddOperator addOp(this->p_context_->device_id_, input.scalar_type());
      addOp.SetSynapseInput(std::move(scOp.GetSynOutputs()[0]));
      auto& add_syn =
          addOp.SetSynapseInput(std::move(p_context_->syn_inputs_[2]));
      // Build Params for the graph
      Scalar alphaValue = 1.0;
      torch::jit::Stack stack;
      stack.emplace_back(IValue(scOp.GetOutputs()[0]));
      stack.emplace_back(IValue(bias));
      stack.emplace_back(IValue(alphaValue));
      addOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
      p_context_->syn_inputs_[2] = std::move(add_syn);
      stack.clear();

      p_context_->syn_outputs_.emplace_back(
          std::move(addOp.GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(std::move(addOp.GetOutputs()[0]));
    }
  }
}

void ConvOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  at::Tensor input = inputs[0].toTensor();
  at::Tensor weight = inputs[1].toTensor();
  const auto stride = inputs[3].toIntList().vec();
  const auto padding = inputs[4].toIntList().vec();
  const bool transposed = inputs[6].toBool();

  c10::MemoryFormat memory_format =
      habana_helpers::get_memory_format({&input, &weight});

  auto shape_out = compute_output_shape(
      input.sizes().vec(),
      weight.sizes().vec(),
      padding,
      stride,
      false,
      transposed,
      c10::MemoryFormat::ChannelsLast);

  auto output = at::empty(shape_out, input.options(), memory_format);
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
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
  int64_t pos_in[] = {0, 2, 3, 1};
  int64_t pos_w[] = {2, 3, 1, 0};
  std::vector<const at::Tensor*> pt_in{&input};
  std::vector<at::Tensor*> pt_out{&input_nhwc};
  IntArrayRef new_dim_pos_in = pos_in;
  IntArrayRef new_dim_pos_w = pos_w;
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
  int64_t pos_out[] = {0, 3, 1, 2};
  IntArrayRef new_dim_pos_out = pos_out;
  pt_new_pos = {&new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  PT_KERNEL_END;
  return output;
}

static auto& KernelRegistry = habana::KernelRegistry().add(
    "aten::convolution_overrideable",
    [](const int device_id, c10::ScalarType node_type) {
      return std::make_shared<ConvOperator>(device_id, node_type);
    });

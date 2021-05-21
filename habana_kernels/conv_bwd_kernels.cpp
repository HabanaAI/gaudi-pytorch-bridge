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
#include "habana_kernels/conv_bwd_kernels.h"
#include "habana_kernels/conv_kernels.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "kernel_utils.h"

using namespace torch;

extern synConvolutionParams synapse_conv_params_builder(
    const IntArrayRef& weight, // HWCK
    const IntArrayRef& stride, // HW
    const IntArrayRef& padding, // HW
    const IntArrayRef& dilation, // HW
    int64_t groups);

void ConvInputDifferentiationOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
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
      inputs[6].isIntList(),
      "Input arg7 expected to be IntList for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[7].isBoolList(),
      "Input arg8 expected to be BoolList for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[8].isTensor(),
      "Input arg9 expected to be tensor for ConvInputDifferentiation operator");
  TORCH_CHECK(
      inputs[9].isInt(),
      "Input arg10 expected to be Int for ConvInputDifferentiation operator");

  auto grad_out_nhwc = inputs[0].toTensor();
  auto input_nhwc = inputs[1].toTensor();
  auto weight_hwck = inputs[2].toTensor();
  const auto stride = inputs[3].toIntList().vec();
  const auto padding = inputs[4].toIntList().vec();
  const auto dilation = inputs[5].toIntList().vec();
  auto output_padding = inputs[6].toIntList();
  auto output_mask_in = inputs[7].toBoolList();
  auto grad_input_nhwc = inputs[8].toTensor();
  auto groups = inputs[9].toInt();

  synConvolutionParams syn_params = synapse_conv_params_builder(
      weight_hwck.sizes(),
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(dilation),
      groups);

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
      inputs.size() == 10,
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
  TORCH_CHECK(
      inputs[9].isInt(),
      "Input arg10 expected to be Int for ConvInputDifferentiation operator");

  auto grad_out_nhwc = inputs[0].toTensor();
  auto input_nhwc = inputs[1].toTensor();
  auto weight_hwck = inputs[2].toTensor();
  const auto stride = inputs[3].toIntList().vec();
  const auto padding = inputs[4].toIntList().vec();
  const auto dilation = inputs[5].toIntList().vec();
  auto output_padding = inputs[6].toIntList();
  auto output_mask_in = inputs[7].toBoolList();
  auto grad_weight = inputs[8].toTensor();
  auto groups = inputs[9].toInt();

  synConvolutionParams syn_params = synapse_conv_params_builder(
      grad_weight.sizes(),
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(dilation),
      groups);

  p_context_->params_.emplace<synConvolutionParams>(syn_params);
  p_context_->params_size_ = sizeof(syn_params);

  AllocateSynapseOutput(graph, grad_weight, is_output_persistent);
  AddNodeToSynapseGraph(graph, &syn_params, sizeof(syn_params));
}

void ConvBackwardOperator::ComputeBiasGrad(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent,
    bool mask_grad_in) {
  auto grad_out_nhwc = inputs[0].toTensor();
  auto grad_bias = habana_helpers::createPTTensor(
      grad_out_nhwc,
      {grad_out_nhwc.size(3)},
      grad_out_nhwc.options(),
      c10::nullopt,
      is_output_persistent[2]);

  if (mask_grad_in) {
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
    SumOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent[2]);

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

void ConvBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 10,
      "Incorrect size of inputs expected for ConvBackward operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for ConvBackward operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for ConvBackward operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be tensor for ConvBackward operator");
  TORCH_CHECK(
      inputs[3].isIntList(),
      "Input arg4 expected to be IntList for ConvBackward operator");
  TORCH_CHECK(
      inputs[4].isIntList(),
      "Input arg5 expected to be IntList for ConvBackward operator");
  TORCH_CHECK(
      inputs[5].isIntList(),
      "Input arg6 expected to be IntList for ConvBackward operator");
  TORCH_CHECK(
      inputs[6].isBool(),
      "Input arg7 expected to be Bool for ConvBackward operator");
  TORCH_CHECK(
      inputs[7].isIntList(),
      "Input arg8 expected to be IntList for ConvBackward operator");
  TORCH_CHECK(
      inputs[8].isInt(),
      "Input arg9 expected to be Int for ConvBackward operator");
  TORCH_CHECK(
      inputs[9].isBoolList(),
      "Input arg10 expected to be BoolList for ConvBackward operator");
  TORCH_CHECK(
      is_output_persistent.size() == 3,
      "ConvBackwardOperator: #is_output_persistent should be 3");

  auto grad_out_nhwc = inputs[0].toTensor();
  auto input_nhwc = inputs[1].toTensor();
  auto weight_hwck = inputs[2].toTensor();
  const auto stride = inputs[3].toIntList().vec();
  const auto padding = inputs[4].toIntList().vec();
  const auto dilation = inputs[5].toIntList().vec();
  auto transposed = inputs[6].toBool();
  auto output_padding = inputs[7].toIntList().vec();
  auto groups = inputs[8].toInt();
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

  if (transposed) { // conv_transpose2d bwd
    // Create the "spatial_convolution" operator
    ConvOperator ConvInputDiffOp(
        this->p_context_->device_id_, grad_out_nhwc.scalar_type());
    if (output_mask_in[0]) {
      // Assign Inputs to the Operator
      auto& grad_out_nhwc_syn = ConvInputDiffOp.SetSynapseInput(
          std::move(p_context_->syn_inputs_[0]));
      auto& weight_hwck_syn = ConvInputDiffOp.SetSynapseInput(
          std::move(p_context_->syn_inputs_[2]));

      // Build Params for the graph
      // use spatial_convolution with bias = None and
      // transposed = false (since we want to use "spatial_convolution" guid)
      Tensor bias = Tensor();
      std::vector<c10::IValue> stack = {
          IValue(grad_out_nhwc),
          IValue(weight_hwck),
          IValue(bias),
          IValue(stride),
          IValue(padding),
          IValue(dilation),
          IValue(false),
          IValue(output_padding),
          IValue(groups)};
      ConvInputDiffOp.AllocateAndAddSynapseNode(
          graph, stack, is_output_persistent[0]);

      p_context_->syn_inputs_[0] = std::move(grad_out_nhwc_syn);
      p_context_->syn_inputs_[2] = std::move(weight_hwck_syn);

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
      // Create the operator
      std::string node_type = "dedw";
      ConvWeightDifferentiationOperator ConvWeightDiffOp(
          this->p_context_->device_id_, node_type);

      // Assign Inputs to the Operator
      auto& input_nhwc_syn = ConvWeightDiffOp.SetSynapseInput(
          std::move(p_context_->syn_inputs_[1]));
      auto& grad_out_nhwc_syn = ConvWeightDiffOp.SetSynapseInput(
          std::move(p_context_->syn_inputs_[0]));

      // Build Params for the graph
      // order of input_nhwc & grad_out_nhwc swapped (w.r.t. regular
      // convolution backward weight gradient computation)
      std::vector<c10::IValue> stack = {
          IValue(input_nhwc),
          IValue(grad_out_nhwc),
          IValue(weight_hwck),
          IValue(stride),
          IValue(padding),
          IValue(dilation),
          IValue(output_padding),
          IValue(output_mask_in),
          IValue(grad_weight),
          IValue(groups)};
      ConvWeightDiffOp.AllocateAndAddSynapseNode(
          graph, stack, is_output_persistent[1]);

      p_context_->syn_inputs_[1] = std::move(input_nhwc_syn);
      p_context_->syn_inputs_[0] = std::move(grad_out_nhwc_syn);
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

  } else { // conv2d backwards
    // Add "dedw" node followed by "dedx" node. Adding in reverse order causes a
    // simulator crash (TBD: investigate later if required)

    // Create the operator
    std::string node_type = "dedw";
    ConvWeightDifferentiationOperator ConvWeightDiffOp(
        this->p_context_->device_id_, node_type);
    if (output_mask_in[1]) {
      // Assign Inputs to the Operator
      auto& grad_out_nhwc_syn = ConvWeightDiffOp.SetSynapseInput(
          std::move(p_context_->syn_inputs_[0]));
      auto& input_nhwc_syn = ConvWeightDiffOp.SetSynapseInput(
          std::move(p_context_->syn_inputs_[1]));

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
          IValue(grad_weight),
          IValue(groups)};
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
      auto& grad_out_nhwc_syn = ConvInputDiffOp.SetSynapseInput(
          std::move(p_context_->syn_inputs_[0]));
      auto& weight_hwck_syn = ConvInputDiffOp.SetSynapseInput(
          std::move(p_context_->syn_inputs_[2]));

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
          IValue(grad_input_nhwc),
          IValue(groups)};
      ConvInputDiffOp.AllocateAndAddSynapseNode(
          graph, stack, is_output_persistent[0]);

      p_context_->syn_inputs_[0] = std::move(grad_out_nhwc_syn);
      p_context_->syn_inputs_[2] = std::move(weight_hwck_syn);
    }

    // Although we have "dedw" node first in the graph followed by "dedx", when
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
  }

  // Bias grad computation same for conv2d bwd and conv2d_transpose bwd
  ComputeBiasGrad(graph, inputs, is_output_persistent, output_mask_in[2]);
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
      // conv weights are HWCK whereas conv2d weights are HWKC
      transposed ? 3 : 2);

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

  if (!transposed) {
    // this is just checking size computation for conv_fwd again
    TORCH_CHECK(
        output_H ==
        habana_helpers::compute_output_size(
            input_H, pad_H, filter_H, stride_H, false, false));
    TORCH_CHECK(
        output_W ==
        habana_helpers::compute_output_size(
            input_W, pad_W, filter_W, stride_W, false, false));
  } else {
    // this is checking size computation for conv_tranpose2d bwd
    // which uses conv_fwd, where grad_output is input and grad_in
    // (same size as input) is output
    TORCH_CHECK(
        input_H ==
        habana_helpers::compute_output_size(
            output_H, pad_H, filter_H, stride_H, false, false));
    TORCH_CHECK(
        input_W ==
        habana_helpers::compute_output_size(
            output_W, pad_W, filter_W, stride_W, false, false));
  }

  // convert tensors to synapse memory format
  Tensor input_nhwc = input;
  Tensor grad_out_nhwc = grad_output;
  Tensor weight_hwck = weight;
  std::vector<const at::Tensor*> pt_in{&input, &grad_output};
  std::vector<at::Tensor*> pt_out{&input_nhwc, &grad_out_nhwc};
  int64_t dim_pos_in[] = {0, 2, 3, 1};
  int64_t dim_grad_out[] = {0, 2, 3, 1};
  int64_t dim_pos_w[] = {2, 3, 1, 0};
  IntArrayRef new_dim_pos_in = dim_pos_in;
  IntArrayRef new_dim_pos_grad_out = dim_grad_out;
  IntArrayRef new_dim_pos_w = dim_pos_w;
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
    int64_t dim_pos_out[] = {0, 3, 1, 2};
    IntArrayRef new_dim_pos_out = dim_pos_out;
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

static auto& KernelRegistry = habana::KernelRegistry().add(
    "aten::convolution_backward_overrideable",
    [](const int device_id, c10::ScalarType node_type) {
      return std::make_shared<ConvBackwardOperator>(device_id, node_type);
    });

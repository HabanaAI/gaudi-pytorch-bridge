/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include <ATen/InferSize.h>
#include <synapse_api.h>
#include <torch/script.h>
#include <iostream>
#include <string>

#include "backend/backend_meta.h"
#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "conv_pool_utils.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/conv_bwd_kernels.h"
#include "habana_kernels/conv_kernels.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "kernel_utils.h"

using namespace torch;
using namespace habana;
using namespace synapse_helpers::layouts;

extern bool is_5d_tensor(const std::vector<at::Tensor>& inputs);

extern synConvolution3DParams synapse_conv3d_params_builder(
    const IntArrayRef& weight, // DHWCK
    const IntArrayRef& stride, // DHW
    const IntArrayRef& padding, // DHW
    const IntArrayRef& dilation, // DHW
    int64_t groups);

extern synConvolutionParams synapse_conv_params_builder(
    const IntArrayRef& weight, // HWCK
    const IntArrayRef& stride, // HW
    const IntArrayRef& padding, // HW
    const IntArrayRef& dilation, // HW
    int64_t groups);

OutputShapeInfRetType Conv3dInputDifferentiationOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto grad_input_nhwc = inputs[8].toTensor();

  OutputShapeInfRetType out;
  auto tensor_meta_data = TensorMetaData(
      grad_input_nhwc.sizes().vec(),
      HabanaOperator::CalculateStrides(
          grad_input_nhwc.sizes(), grad_input_nhwc.suggest_memory_format()),
      grad_input_nhwc.scalar_type(),
      grad_input_nhwc.suggest_memory_format());
  out.AddShapeTensor(tensor_meta_data);
  out.AddOutputTensor(tensor_meta_data);
  return out;
}

void Conv3dInputDifferentiationOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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

  synConvolution3DParams syn_params = synapse_conv3d_params_builder(
      weight_hwck.sizes(),
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(dilation),
      groups);

  p_context_->params_.emplace<synConvolution3DParams>(syn_params);
  p_context_->params_size_ = sizeof(syn_params);

  // Allocate Shape Tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, grad_input_nhwc);
  }

  AllocateSynapseOutput(graph, grad_input_nhwc, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &syn_params, sizeof(syn_params));
}

OutputShapeInfRetType ConvInputDifferentiationOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto grad_input_nhwc = inputs[8].toTensor();

  OutputShapeInfRetType out;
  auto tensor_meta_data = TensorMetaData(
      grad_input_nhwc.sizes().vec(),
      HabanaOperator::CalculateStrides(
          grad_input_nhwc.sizes(), grad_input_nhwc.suggest_memory_format()),
      grad_input_nhwc.scalar_type(),
      grad_input_nhwc.suggest_memory_format());
  out.AddShapeTensor(tensor_meta_data);
  out.AddOutputTensor(tensor_meta_data);
  return out;
}

void ConvInputDifferentiationOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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

  // Allocate Shape Tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, grad_input_nhwc);
  }

  AllocateSynapseOutput(graph, grad_input_nhwc, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &syn_params, sizeof(syn_params));
}

OutputShapeInfRetType Conv3dWeightDifferentiationOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto grad_weight = inputs[8].toTensor();

  OutputShapeInfRetType out;
  out.AddOutputTensor(TensorMetaData(
      grad_weight.sizes().vec(),
      HabanaOperator::CalculateStrides(
          grad_weight.sizes(), grad_weight.suggest_memory_format()),
      grad_weight.scalar_type(),
      grad_weight.suggest_memory_format()));
  return out;
}

void Conv3dWeightDifferentiationOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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

  synConvolution3DParams syn_params = synapse_conv3d_params_builder(
      grad_weight.sizes(),
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(dilation),
      groups);

  p_context_->params_.emplace<synConvolution3DParams>(syn_params);
  p_context_->params_size_ = sizeof(syn_params);

  AllocateSynapseOutput(graph, grad_weight, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &syn_params, sizeof(syn_params));
}

OutputShapeInfRetType ConvWeightDifferentiationOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto grad_weight = inputs[8].toTensor();

  OutputShapeInfRetType out;
  out.AddOutputTensor(TensorMetaData(
      grad_weight.sizes().vec(),
      HabanaOperator::CalculateStrides(
          grad_weight.sizes(), grad_weight.suggest_memory_format()),
      grad_weight.scalar_type(),
      grad_weight.suggest_memory_format()));
  return out;
}

void ConvWeightDifferentiationOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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

  AllocateSynapseOutput(graph, grad_weight, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &syn_params, sizeof(syn_params));
}

void ConvBackwardOperator::ComputeBiasGrad3d(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata,
    bool mask_grad_in) {
  auto grad_out_nhwc = inputs[0].toTensor();
  auto channel_dim = INPUT_3D_C_IDX;
  std::vector<OutputMetaData> out_2_metadata =
      SelectVectorIndices(output_metadata, {2});
  auto grad_bias = habana::createPTTensor(
      grad_out_nhwc,
      {grad_out_nhwc.size(channel_dim)},
      grad_out_nhwc.options(),
      c10::nullopt,
      out_2_metadata.at(0).persistent);

  if (mask_grad_in) {
    std::vector<int64_t> dim_to_reduce;
    for (int64_t i = 0; i < grad_out_nhwc.ndimension(); ++i) {
      if (i != channel_dim) // skip C dimension
        dim_to_reduce.push_back(i);
    }
    c10::IntArrayRef shape(dim_to_reduce.data(), dim_to_reduce.size());

    at::ScalarType scalar_type = grad_out_nhwc.scalar_type();
    std::string node_type =
        "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

    // Create the operator
    auto SumOp = make_operator<SumDimOutOperator>(
        this->p_context_->device_id_, scalar_type);

    SumOp->SetSynapseInput(p_context_->syn_inputs_[0]);

    std::vector<c10::IValue> stack = {
        IValue(grad_out_nhwc),
        IValue(shape),
        IValue(false),
        IValue(scalar_type),
        IValue(grad_bias)};
    SumOp->AllocateAndAddSynapseNode(graph, stack, out_2_metadata);

    synapse_helpers::tensor& bias_syn_tensor = SumOp->GetSynOutputs()[0];

    p_context_->syn_outputs_.emplace_back(std::move(bias_syn_tensor));
    p_context_->pt_outputs_.emplace_back(std::move(SumOp->GetOutputs()[0]));

  } else {
    p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
        grad_bias,
        graph,
        out_2_metadata.at(0).persistent,
        out_2_metadata.at(0).external,
        c10::nullopt,
        out_2_metadata.at(0).name));
    p_context_->pt_outputs_.emplace_back(grad_bias);
  }
}

void ConvBackwardOperator::ComputeBiasGrad(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata,
    bool mask_grad_in) {
  std::vector<OutputMetaData> out_2_metadata =
      SelectVectorIndices(output_metadata, {2});
  auto channel_dim = INPUT_C_IDX;
  auto grad_out_nhwc = inputs[0].toTensor();
  auto grad_bias = habana::createPTTensor(
      grad_out_nhwc,
      {grad_out_nhwc.size(channel_dim)},
      grad_out_nhwc.options(),
      c10::nullopt,
      out_2_metadata.at(0).persistent);

  if (mask_grad_in) {
    std::vector<int64_t> dim_to_reduce;
    for (int64_t i = 0; i < grad_out_nhwc.ndimension(); ++i) {
      if (i != channel_dim) // skip C dimension
        dim_to_reduce.push_back(i);
    }
    c10::IntArrayRef shape(dim_to_reduce.data(), dim_to_reduce.size());

    at::ScalarType scalar_type = grad_out_nhwc.scalar_type();
    std::string node_type =
        "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

    // Create the operator
    auto SumOp = make_operator<SumDimOutOperator>(
        this->p_context_->device_id_, scalar_type);

    SumOp->SetSynapseInput(p_context_->syn_inputs_[0]);

    std::vector<c10::IValue> stack = {
        IValue(grad_out_nhwc),
        IValue(shape),
        IValue(false),
        IValue(scalar_type),
        IValue(grad_bias)};
    SumOp->AllocateAndAddSynapseNode(graph, stack, out_2_metadata);

    synapse_helpers::tensor& bias_syn_tensor = SumOp->GetSynOutputs()[0];

    p_context_->syn_outputs_.emplace_back(std::move(bias_syn_tensor));
    p_context_->pt_outputs_.emplace_back(std::move(SumOp->GetOutputs()[0]));

  } else {
    p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
        grad_bias,
        graph,
        out_2_metadata.at(0).persistent,
        out_2_metadata.at(0).external,
        c10::nullopt,
        out_2_metadata.at(0).name));
    p_context_->pt_outputs_.emplace_back(grad_bias);
  }
}

OutputShapeInfRetType ConvBackwardOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
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

  std::vector<at::Tensor> temp_inputs{input_nhwc, weight_hwck};
  auto is_conv_3d = is_5d_tensor(temp_inputs);

  c10::MemoryFormat memory_format = habana_helpers::get_memory_format(
      {&input_nhwc, &grad_out_nhwc, &weight_hwck});

  auto grad_weight = habana::nonPersistentTensor(
      weight_hwck, weight_hwck.sizes(), grad_out_nhwc.options(), memory_format);

  auto grad_input_nhwc = habana::nonPersistentTensor(
      input_nhwc, input_nhwc.sizes(), grad_out_nhwc.options(), memory_format);

  OutputShapeInfRetType out;
  if (transposed) { // conv_transpose2d bwd
    // Create the "spatial_convolution" operator
    auto ConvInputDiffOp = make_operator<ConvOperator>(
        this->p_context_->device_id_, grad_out_nhwc.scalar_type());

    if (output_mask_in[0]) {
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
      auto ConvInputDiffOp_out =
          out.call_ComputeOutputShape(ConvInputDiffOp, stack);
      auto out_tensor = ConvInputDiffOp_out.GetOutputTensor(0);
      out.MoveToOutput(std::move(out_tensor));
    } else {
      out.AddOutputTensor(TensorMetaData(
          input_nhwc.sizes().vec(),
          HabanaOperator::CalculateStrides(input_nhwc.sizes(), memory_format),
          grad_out_nhwc.scalar_type(),
          memory_format));
    }

    if (output_mask_in[1]) {
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

      // Create the operator
      if (is_conv_3d) {
        std::string node_type = "dedw3d";
        auto Conv3dWeightDiffOp =
            make_operator<Conv3dWeightDifferentiationOperator>(
                this->p_context_->device_id_, node_type);
        auto Conv3dWeightDiffOp_out =
            out.call_ComputeOutputShape(Conv3dWeightDiffOp, stack);
        auto out_tensor = Conv3dWeightDiffOp_out.GetOutputTensor(0);
        out.MoveToOutput(std::move(out_tensor));
      } else {
        std::string node_type = "dedw";
        auto ConvWeightDiffOp =
            make_operator<ConvWeightDifferentiationOperator>(
                this->p_context_->device_id_, node_type);
        auto ConvWeightDiffOp_out =
            out.call_ComputeOutputShape(ConvWeightDiffOp, stack);
        auto out_tensor = ConvWeightDiffOp_out.GetOutputTensor(0);
        out.MoveToOutput(std::move(out_tensor));
      }
    } else {
      out.AddOutputTensor(TensorMetaData(
          weight_hwck.sizes().vec(),
          HabanaOperator::CalculateStrides(weight_hwck.sizes(), memory_format),
          grad_out_nhwc.scalar_type(),
          memory_format));
    }
  } else { // conv2d backwards
    // Add "dedw" node followed by "dedx" node. Adding in reverse order causes a
    // simulator crash (TBD: investigate later if required)
    std::vector<c10::IValue> stackDedw = {
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

    std::vector<c10::IValue> stackDedx = {
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

    // Create the operator
    if (is_conv_3d) {
      // Although we have "dedw3d" node first in the graph followed by "dedx3d",
      // when pushing outputs we want to maintain correct order
      if (output_mask_in[0]) {
        auto Conv3dInputDiffOp =
            make_operator<Conv3dInputDifferentiationOperator>(
                this->p_context_->device_id_, "dedx3d");
        auto Conv3dInputDiffOp_out =
            out.call_ComputeOutputShape(Conv3dInputDiffOp, stackDedx);
        auto out_tensor = Conv3dInputDiffOp_out.GetOutputTensor(0);
        out.MoveToOutput(std::move(out_tensor));
      } else {
        out.AddOutputTensor(TensorMetaData(
            input_nhwc.sizes().vec(),
            HabanaOperator::CalculateStrides(input_nhwc.sizes(), memory_format),
            grad_out_nhwc.scalar_type(),
            memory_format));
      }

      if (output_mask_in[1]) {
        auto Conv3dWeightDiffOp =
            make_operator<Conv3dWeightDifferentiationOperator>(
                this->p_context_->device_id_, "dedw3d");
        auto Conv3dWeightDiffOp_out =
            out.call_ComputeOutputShape(Conv3dWeightDiffOp, stackDedw);
        auto out_tensor = Conv3dWeightDiffOp_out.GetOutputTensor(0);
        out.MoveToOutput(std::move(out_tensor));
      } else {
        out.AddOutputTensor(TensorMetaData(
            weight_hwck.sizes().vec(),
            HabanaOperator::CalculateStrides(
                weight_hwck.sizes(), memory_format),
            grad_out_nhwc.scalar_type(),
            memory_format));
      }
    } else {
      // Although we have "dedw" node first in the graph followed by "dedx",
      // when pushing outputs we want to maintain correct order
      OutputShapeInfRetType ConvWeightDiffOp_out;
      OutputShapeInfRetType ConvInputDiffOp_out;
      if (output_mask_in[1]) {
        auto ConvWeightDiffOp =
            make_operator<ConvWeightDifferentiationOperator>(
                this->p_context_->device_id_, "dedw");
        ConvWeightDiffOp_out =
            out.call_ComputeOutputShape(ConvWeightDiffOp, stackDedw);
      }

      if (output_mask_in[0]) {
        auto ConvInputDiffOp = make_operator<ConvInputDifferentiationOperator>(
            this->p_context_->device_id_, "dedx");
        ConvInputDiffOp_out =
            out.call_ComputeOutputShape(ConvInputDiffOp, stackDedx);
      }

      if (output_mask_in[0]) {
        auto out_tensor = ConvInputDiffOp_out.GetOutputTensor(0);
        out.MoveToOutput(std::move(out_tensor));
      } else {
        out.AddOutputTensor(TensorMetaData(
            input_nhwc.sizes().vec(),
            HabanaOperator::CalculateStrides(input_nhwc.sizes(), memory_format),
            grad_out_nhwc.scalar_type(),
            memory_format));
      }
      if (output_mask_in[1]) {
        auto out_tensor = ConvWeightDiffOp_out.GetOutputTensor(0);
        out.MoveToOutput(std::move(out_tensor));
      } else {
        out.AddOutputTensor(TensorMetaData(
            weight_hwck.sizes().vec(),
            HabanaOperator::CalculateStrides(
                weight_hwck.sizes(), memory_format),
            grad_out_nhwc.scalar_type(),
            memory_format));
      }
    }
  }

  // Bias grad computation same for conv2d bwd and conv2d_transpose bwd
  int64_t channel_dim;
  if (is_conv_3d) {
    channel_dim = INPUT_3D_C_IDX;
  } else {
    channel_dim = INPUT_C_IDX;
  }

  auto grad_bias = habana::nonPersistentTensor(
      grad_out_nhwc,
      {grad_out_nhwc.size(channel_dim)},
      grad_out_nhwc.options(),
      c10::nullopt);

  if (output_mask_in[2]) {
    std::vector<int64_t> dim_to_reduce;
    for (int64_t i = 0; i < grad_out_nhwc.ndimension(); ++i) {
      if (i != channel_dim) // skip C dimension
        dim_to_reduce.push_back(i);
    }
    c10::IntArrayRef shape(dim_to_reduce.data(), dim_to_reduce.size());

    at::ScalarType scalar_type = grad_out_nhwc.scalar_type();

    // Create the operator
    auto SumOp = make_operator<SumDimOutOperator>(
        this->p_context_->device_id_, scalar_type);
    std::vector<c10::IValue> stack2 = {
        IValue(grad_out_nhwc),
        IValue(shape),
        IValue(false),
        IValue(scalar_type),
        IValue(grad_bias)};
    auto SumOp_out = out.call_ComputeOutputShape(SumOp, stack2);
    auto out_tensor = SumOp_out.GetOutputTensor(0);
    out.MoveToOutput(std::move(out_tensor));
  } else {
    out.AddOutputTensor(TensorMetaData(
        grad_bias.sizes().vec(),
        HabanaOperator::CalculateStrides(grad_bias.sizes(), memory_format),
        grad_out_nhwc.scalar_type(),
        memory_format));
  }

  return out;
}

void ConvBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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
      output_metadata.size() == 3,
      "ConvBackwardOperator: #output_metadata should be 3");

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

  std::vector<at::Tensor> temp_inputs{input_nhwc, weight_hwck};
  auto is_conv_3d = is_5d_tensor(temp_inputs);

  c10::MemoryFormat memory_format = habana_helpers::get_memory_format(
      {&input_nhwc, &grad_out_nhwc, &weight_hwck});

  auto grad_weight = habana::createPTTensor(
      weight_hwck,
      weight_hwck.sizes(),
      grad_out_nhwc.options(),
      memory_format,
      output_metadata.at(1).persistent);
  if (output_metadata.at(1).persistent) {
    // set Weights layout HWCK
    auto hb_grad_weight{get_tensor_extra_meta(grad_weight, true)};
    if (hb_grad_weight)
      hb_grad_weight->set_tensor_layout(habana::LayoutFormat::HWCK);
  }
  auto grad_input_nhwc = habana::createPTTensor(
      input_nhwc,
      input_nhwc.sizes(),
      grad_out_nhwc.options(),
      memory_format,
      output_metadata.at(0).persistent);

  if (transposed) { // conv_transpose2d bwd
    // Create the "spatial_convolution" operator
    auto ConvInputDiffOp = make_operator<ConvOperator>(
        this->p_context_->device_id_, grad_out_nhwc.scalar_type());
    if (output_mask_in[0]) {
      // Assign Inputs to the Operator
      ConvInputDiffOp->SetSynapseInput(p_context_->syn_inputs_[0]);
      ConvInputDiffOp->SetSynapseInput(p_context_->syn_inputs_[2]);

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
      ConvInputDiffOp->AllocateAndAddSynapseNode(
          graph, stack, {output_metadata.at(0)});

      synapse_helpers::tensor& grad_in_nhwc_syn_tensor =
          ConvInputDiffOp->GetSynOutputs()[0];
      p_context_->syn_outputs_.emplace_back(std::move(grad_in_nhwc_syn_tensor));
      p_context_->pt_outputs_.emplace_back(
          std::move(ConvInputDiffOp->GetOutputs()[0]));
    } else {
      p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
          grad_input_nhwc,
          graph,
          output_metadata.at(0).persistent,
          output_metadata.at(0).external,
          c10::nullopt,
          output_metadata.at(0).name));
      p_context_->pt_outputs_.emplace_back(grad_input_nhwc);
    }

    if (output_mask_in[1]) {
      auto populateOp = [&](std::shared_ptr<habana::HabanaOperator>
                                ConvWeightDiffOp) mutable {
        // Assign Inputs to the Operator
        ConvWeightDiffOp->SetSynapseInput(p_context_->syn_inputs_[1]);
        ConvWeightDiffOp->SetSynapseInput(p_context_->syn_inputs_[0]);

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
        ConvWeightDiffOp->AllocateAndAddSynapseNode(
            graph, stack, {output_metadata.at(1)});

        synapse_helpers::tensor& grad_weight_syn_tensor =
            ConvWeightDiffOp->GetSynOutputs()[0];

        p_context_->syn_outputs_.emplace_back(
            std::move(grad_weight_syn_tensor));
        p_context_->pt_outputs_.emplace_back(
            std::move(ConvWeightDiffOp->GetOutputs()[0]));
      };
      // Create the operator
      if (is_conv_3d) {
        std::string node_type = "dedw3d";
        auto ConvWeightDiffOp =
            make_operator<Conv3dWeightDifferentiationOperator>(
                this->p_context_->device_id_, node_type);

        populateOp(ConvWeightDiffOp);
      } else {
        std::string node_type = "dedw";
        auto ConvWeightDiffOp =
            make_operator<ConvWeightDifferentiationOperator>(
                this->p_context_->device_id_, node_type);

        populateOp(ConvWeightDiffOp);
      }
    } else {
      p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
          grad_weight,
          graph,
          output_metadata.at(1).persistent,
          output_metadata.at(1).external,
          c10::nullopt));
      p_context_->pt_outputs_.emplace_back(grad_weight);
    }

  } else { // conv2d backwards
    // Add "dedw" node followed by "dedx" node. Adding in reverse order causes a
    // simulator crash (TBD: investigate later if required)

    auto populateDedwOp =
        [&](std::shared_ptr<habana::HabanaOperator>& ConvWeightDiffOp) mutable {
          if (output_mask_in[1]) {
            if (is_conv_3d) {
              ConvWeightDiffOp =
                  make_operator<Conv3dWeightDifferentiationOperator>(
                      this->p_context_->device_id_, "dedw3d");
            } else {
              ConvWeightDiffOp =
                  make_operator<ConvWeightDifferentiationOperator>(
                      this->p_context_->device_id_, "dedw");
            }
            // Assign Inputs to the Operator
            ConvWeightDiffOp->SetSynapseInput(p_context_->syn_inputs_[0]);
            ConvWeightDiffOp->SetSynapseInput(p_context_->syn_inputs_[1]);

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
            ConvWeightDiffOp->AllocateAndAddSynapseNode(
                graph, stack, {output_metadata.at(1)});
          }
        };
    auto populateDedxOp =
        [&](std::shared_ptr<habana::HabanaOperator>& ConvInputDiffOp) mutable {
          if (output_mask_in[0]) {
            if (is_conv_3d) {
              ConvInputDiffOp =
                  make_operator<Conv3dInputDifferentiationOperator>(
                      this->p_context_->device_id_, "dedx3d");
            } else {
              ConvInputDiffOp = make_operator<ConvInputDifferentiationOperator>(
                  this->p_context_->device_id_, "dedx");
            }
            // Assign Inputs to the Operator
            ConvInputDiffOp->SetSynapseInput(p_context_->syn_inputs_[0]);
            ConvInputDiffOp->SetSynapseInput(p_context_->syn_inputs_[2]);

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
            ConvInputDiffOp->AllocateAndAddSynapseNode(
                graph, stack, {output_metadata.at(0)});
          }
        };
    auto reverseOrderOp =
        [&](std::shared_ptr<habana::HabanaOperator>& ConvInputDiffOp,
            std::shared_ptr<habana::HabanaOperator>& ConvWeightDiffOp) mutable {
          // Although we have "dedw3d" node first in the graph followed by
          // "dedx3d", when pushing outputs we want to maintain correct order
          if (output_mask_in[0]) {
            synapse_helpers::tensor& grad_in_nhwc_syn_tensor =
                ConvInputDiffOp->GetSynOutputs()[0];
            p_context_->syn_outputs_.emplace_back(
                std::move(grad_in_nhwc_syn_tensor));
            p_context_->pt_outputs_.emplace_back(
                std::move(ConvInputDiffOp->GetOutputs()[0]));
          } else {
            p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
                grad_input_nhwc,
                graph,
                output_metadata.at(0).persistent,
                output_metadata.at(0).external,
                grad_input_nhwc.scalar_type()));
            p_context_->pt_outputs_.emplace_back(grad_input_nhwc);
          }

          if (output_mask_in[1]) {
            synapse_helpers::tensor& grad_weight_syn_tensor =
                ConvWeightDiffOp->GetSynOutputs()[0];

            p_context_->syn_outputs_.emplace_back(
                std::move(grad_weight_syn_tensor));
            p_context_->pt_outputs_.emplace_back(
                std::move(ConvWeightDiffOp->GetOutputs()[0]));
          } else {
            p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
                grad_weight,
                graph,
                output_metadata.at(1).persistent,
                output_metadata.at(1).external,
                grad_weight.scalar_type()));
            p_context_->pt_outputs_.emplace_back(grad_weight);
          }
        };
    // Create the operator
    std::shared_ptr<habana::HabanaOperator> ConvWeightDiffOp;
    std::shared_ptr<habana::HabanaOperator> ConvInputDiffOp;
    populateDedwOp(ConvWeightDiffOp);
    populateDedxOp(ConvInputDiffOp);

    // Although we have "dedw" node first in the graph followed by "dedx",
    // when pushing outputs we want to maintain correct order
    reverseOrderOp(ConvInputDiffOp, ConvWeightDiffOp);
  }

  // Bias grad computation same for conv2d bwd and conv2d_transpose bwd
  if (is_conv_3d) {
    ComputeBiasGrad3d(graph, inputs, output_metadata, output_mask_in[2]);
  } else {
    ComputeBiasGrad(graph, inputs, output_metadata, output_mask_in[2]);
  }
}

void ConvBackwardOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto grad_out_nhwc = inputs[0].toTensor();
  auto input_nhwc = inputs[1].toTensor();
  auto weight_hwck = inputs[2].toTensor();
  const auto stride = inputs[3].toIntList().vec();
  const auto padding = inputs[4].toIntList().vec();
  const auto dilation = inputs[5].toIntList().vec();
  [[maybe_unused]] auto transposed = inputs[6].toBool();
  auto output_padding = inputs[7].toIntList();
  [[maybe_unused]] auto groups = inputs[8].toInt();
  auto output_mask_in = inputs[9].toBoolList();

  std::vector<at::Tensor> temp_inputs{input_nhwc, weight_hwck};
  auto is_conv_3d = is_5d_tensor(temp_inputs);

  c10::MemoryFormat memory_format = habana_helpers::get_memory_format(
      {&input_nhwc, &grad_out_nhwc, &weight_hwck});

  auto grad_weight =
      at::empty(weight_hwck.sizes(), grad_out_nhwc.options(), memory_format);
  auto grad_input_nhwc =
      at::empty(input_nhwc.sizes(), grad_out_nhwc.options(), memory_format);
  auto channel_dim = is_conv_3d ? 4 : 3;
  auto grad_bias = at::empty(
      {grad_out_nhwc.size(channel_dim)},
      grad_out_nhwc.options(),
      memory_format);

  HabanaOperator::SetPTOutput(grad_input_nhwc);
  HabanaOperator::SetPTOutput(grad_weight);
  HabanaOperator::SetPTOutput(grad_bias);
}

static auto& ConvBwdKernelsKernelRegistry = habana::KernelRegistry().add(
    "aten::convolution_backward_overrideable",
    KERNEL_FN(ConvBackwardOperator));

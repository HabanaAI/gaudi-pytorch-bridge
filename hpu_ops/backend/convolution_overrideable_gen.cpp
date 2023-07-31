/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include <string>
#include "backend/synapse_helpers/layout_utils.h"
#include "generated/backend/convolution.h"
#include "hpu_ops/common/convolution_gen.h"

using namespace synapse_helpers::layouts;

namespace habana {

static std::shared_ptr<void> ConvolutionOverrideable3dParams(
    const at::IntArrayRef& weight, // DHWCK
    const at::IntArrayRef& stride, // DHW
    const at::IntArrayRef& padding, // DHW
    const at::IntArrayRef& dilation, // DHW
    int64_t groups,
    size_t& size) {
  PARAMS_STUB(synConvolution3DParams);
  params->kernel[CONV_KERNEL_DEPTH] = weight[2];
  params->kernel[CONV_KERNEL_HEIGHT] = weight[3];
  params->kernel[CONV_KERNEL_WIDTH] = weight[4];
  params->stride[CONV_STRIDE_DEPTH] = stride[0];
  params->stride[CONV_STRIDE_HEIGHT] = stride[1];
  params->stride[CONV_STRIDE_WIDTH] = stride[2];
  params->dilation[CONV_DIL_DEPTH] = dilation[0];
  params->dilation[CONV_DIL_HEIGHT] = dilation[1];
  params->dilation[CONV_DIL_WIDTH] = dilation[2];
  params->padding[CONV_PAD_FRONT] = padding[0];
  params->padding[CONV_PAD_BACK] = padding[0];
  params->padding[CONV_PAD_TOP] = padding[1];
  params->padding[CONV_PAD_BOTTOM] = padding[1];
  params->padding[CONV_PAD_LEFT] = padding[2];
  params->padding[CONV_PAD_RIGHT] = padding[2];
  params->nGroups = groups;

  return params;
}

static std::shared_ptr<void> ConvolutionOverrideable2dParams(
    const at::IntArrayRef& weight, // HWCK
    const at::IntArrayRef& stride, // HW
    const at::IntArrayRef& padding, // HW
    const at::IntArrayRef& dilation, // HW
    int64_t groups,
    size_t& size) {
  PARAMS_STUB(synConvolutionParams);
  params->dH = stride[0];
  params->dW = stride[1];
  params->kH = weight[2];
  params->kW = weight[3];
  params->dilH = dilation[0];
  params->dilW = dilation[1];
  params->setPadT(padding[0]);
  params->setPadB(padding[0]);
  params->setPadL(padding[1]);
  params->setPadR(padding[1]);
  params->nGroups = groups;

  return params;
}

std::shared_ptr<void> FillConvolutionOverrideableParams(
    const at::Stack& stack,
    size_t& size) {
  auto weight_shape = stack_tensor(stack, 1).sizes().vec();
  auto stride = stack[3].toIntList().vec();
  auto padding = stack[4].toIntList().vec();
  auto dilation = stack[5].toIntList().vec();
  const int64_t groups = stack[8].toInt();

  if (stack_tensor(stack, 0).dim() == 3) {
    weight_shape.push_back(1);
    stride.push_back(1);
    padding.push_back(0);
    dilation.push_back(1);
  }

  if (stack_tensor(stack, 0).dim() == 5) {
    return ConvolutionOverrideable3dParams(
        weight_shape, stride, padding, dilation, groups, size);
  } else {
    return ConvolutionOverrideable2dParams(
        weight_shape, stride, padding, dilation, groups, size);
  }
}

static int64_t ComputeOutputSize(
    const int64_t input_dim,
    const int64_t padding,
    const int64_t dilation,
    const int64_t kernel_size,
    const int64_t stride,
    const int64_t output_padding,
    const bool transposed) {
  if (!transposed) {
    return (input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) /
        stride +
        1;
  } else {
    // conv2d fwd output shape computation done as per formula provided below
    // https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
    return (input_dim - 1) * stride - 2 * padding +
        dilation * (kernel_size - 1) + output_padding + 1;
  }
}

sizes_vec ConvolutionOverrideableOutputShape(const at::Stack& stack) {
  auto shape_in = stack_tensor(stack, 0).sizes();
  auto shape_wt = stack_tensor(stack, 1).sizes();
  const auto stride = stack[3].toIntList().vec();
  const auto padding = stack[4].toIntList().vec();
  const auto dilation = stack[5].toIntList().vec();
  const bool transposed = stack[6].toBool();
  const auto output_padding = stack[7].toIntList().vec();
  const int64_t groups = stack[8].toInt();

  auto K = transposed ? shape_wt[1] * groups : shape_wt[0];
  std::vector<int64_t> out_shape{shape_in[0], K};
  for (int i = 0; i < shape_in.size() - 2; ++i) {
    out_shape.push_back(ComputeOutputSize(
        shape_in[i + 2],
        padding[i],
        dilation[i],
        shape_wt[i + 2],
        stride[i],
        output_padding[i],
        transposed));
  }

  return {out_shape};
}

void ConvolutionOverrideable::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;

  at::Tensor input = stack_tensor(stack, 0);
  at::Tensor weight = stack_tensor(stack, 1);
  auto bias = stack.at(2).toOptional<at::Tensor>().value_or(at::Tensor());
  const bool transposed = stack[6].toBool();

  const bool is_conv_1d = input.dim() == 3;

  // For convolution 1d we add additional reshapes
  IF_CONV1D_RESHAPE_TO_2D(input, 0);
  IF_CONV1D_RESHAPE_TO_2D(weight, 1);

  const uint64_t DIM5 = 5;
  const bool is_conv_3d = input.dim() == DIM5;

  if (is_conv_3d) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
         synapse_helpers::layouts::SynapseLayoutFormat::SRQCK,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN});
  } else {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::SRCK,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  }

  std::string guid = transposed ? "dedx" : "spatial_convolution";
  if (is_conv_3d)
    guid += "3d";

  std::vector<synTensor> inputs = {input_reshaped, weight_reshaped};

  auto out_shape = ConvolutionOverrideableOutputShape(stack)[0];

  if (is_conv_1d) {
    out_shape.push_back(1);
  }

  if (guid == "dedx" || guid == "dedx3d") {
    this->CreateShapeTensorInput(graph, this->ScalarType(), out_shape, inputs);
  } else if (bias.defined()) {
    inputs.emplace_back(syn_in(2));
  }

  const auto& params = FillConvolutionOverrideableParams(stack, size);

  NodeAttr::NodeOutputAttr node_output_attr = {out_shape, ScalarType(), 0};
  if ((transposed && bias.defined()) || is_conv_1d)
    node_output_attr.final_result_index = c10::nullopt;

  auto convOp =
      BuildOp(graph, guid, inputs, {node_output_attr}, params.get(), size);

  SetSynapseLayouts({}, {});

  if (transposed && bias.defined()) {
    // Reshape bias to match to NCHW output format
    int64_t data[5] = {1, bias.sizes().vec()[0], 1, 1, 1};
    c10::IntArrayRef shape(data, is_conv_3d ? 5 : 4);
    synapse_helpers::tensor biasReshaped =
        BuildReshape(this, graph, syn_in(2), shape, ScalarType());

    c10::optional<int> final_result_index_0 =
        is_conv_1d ? c10::optional<int>{c10::nullopt} : c10::optional<int>{0};
    auto addOp = BuildOp(
        graph,
        get_guid_with_precision("add_fwd", ScalarType()),
        {convOp[0].get(), biasReshaped.get()},
        {{out_shape, ScalarType(), final_result_index_0}});

    IF_CONV1D_RESHAPE_TO_ORIG_AND_SET_OUT(addOp[0], out_shape, 0);
  } else {
    IF_CONV1D_RESHAPE_TO_ORIG_AND_SET_OUT(convOp[0], out_shape, 0);
  }
}
} // namespace habana

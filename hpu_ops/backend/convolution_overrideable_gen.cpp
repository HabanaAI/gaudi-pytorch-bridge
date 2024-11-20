/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include <string>
#include "backend/synapse_helpers/layout_utils.h"
#include "generated/backend/convolution_overrideable.h"
#include "hpu_ops/common/convolution_gen.h"

using namespace synapse_helpers::layouts;

namespace habana {

using SynapseLayouts =
    std::vector<synapse_helpers::layouts::SynapseLayoutFormat>;

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

OutputMetaDataVector ConvolutionOverrideableMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto shapeIn = self.sizes();
  auto shapeWt = stack_tensor(stack, 1).sizes();
  const auto stride = stack.at(3).toIntList().vec();
  const auto padding = stack.at(4).toIntList().vec();
  const auto dilation = stack.at(5).toIntList().vec();
  const bool transposed = stack.at(6).toBool();
  const auto outputPadding = stack.at(7).toIntList().vec();
  const int64_t groups = stack.at(8).toInt();

  auto K = transposed ? shapeWt[1] * groups : shapeWt[0];
  std::vector<int64_t> outputShape{shapeIn[0], K};
  for (size_t i = 0; i < shapeIn.size() - 2; ++i) {
    outputShape.push_back(ComputeOutputSize(
        shapeIn[i + 2],
        padding[i],
        dilation[i],
        shapeWt[i + 2],
        stride[i],
        outputPadding[i],
        transposed));
  }

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = outputShape;
  return {meta};
}

static std::pair<SynapseLayouts, SynapseLayouts> MakeLayouts(
    bool is_conv_3d,
    bool transposed) {
  const auto input_layout = is_conv_3d
      ? synapse_helpers::layouts::SynapseLayoutFormat::WHDCN
      : synapse_helpers::layouts::SynapseLayoutFormat::WHCN;
  const auto weight_layout = is_conv_3d
      ? synapse_helpers::layouts::SynapseLayoutFormat::SRQCK
      : synapse_helpers::layouts::SynapseLayoutFormat::SRCK;

  SynapseLayouts in_layouts{input_layout, weight_layout};
  SynapseLayouts out_layouts{input_layout};

  in_layouts.push_back(
      transposed ? input_layout
                 : synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE);

  return std::make_pair(in_layouts, out_layouts);
}

SharedMetaDataVector ConvolutionSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& input = stack.at(0).toTensor();
  const auto& weight = stack.at(1).toTensor();
  const auto& bias =
      stack.at(2).toOptional<at::Tensor>().value_or(at::Tensor());
  const bool biasDefined = bias.defined();
  const bool transposed = stack[6].toBool();
  auto inputRank = input.dim();
  auto weightRank = weight.dim();
  const auto inputDtype = input.scalar_type();
  const auto weightDtype = weight.scalar_type();

  SharedMetaDataVector convolutionSharedMeta;

  const bool is_conv_1d = inputRank == 3;
  if (is_conv_1d) {
    SharedMetaData expandInputDimsMeta("expand_dims");
    expandInputDimsMeta.inputs_data.emplace_back(inputRank, inputDtype);
    expandInputDimsMeta.outputs_data.emplace_back(++inputRank, inputDtype);
    convolutionSharedMeta.push_back(expandInputDimsMeta);

    SharedMetaData expandWeightDimsMeta("expand_dims");
    expandWeightDimsMeta.inputs_data.emplace_back(weightRank, weightDtype);
    expandWeightDimsMeta.outputs_data.emplace_back(++weightRank, weightDtype);
    convolutionSharedMeta.push_back(expandWeightDimsMeta);
  }

  const bool is_conv_3d = inputRank == 5;
  std::string guid = transposed ? "dedx" : "spatial_convolution";
  if (is_conv_3d)
    guid += "3d";

  SharedMetaData convMeta(guid);
  convMeta.inputs_data.emplace_back(inputRank, inputDtype);
  convMeta.inputs_data.emplace_back(weightRank, weightDtype);
  if (biasDefined && !transposed) {
    convMeta.inputs_data.emplace_back(1, bias.scalar_type());
  }
  convMeta.outputs_data.emplace_back(inputRank, inputDtype);
  convolutionSharedMeta.push_back(convMeta);

  if (biasDefined && transposed) {
    SharedMetaData expandMeta("expand_multi_dims");
    expandMeta.inputs_data.emplace_back(1, bias.scalar_type());
    expandMeta.outputs_data.emplace_back(
        is_conv_3d ? 5 : 4, bias.scalar_type());
    convolutionSharedMeta.push_back(expandMeta);

    SharedMetaData addMeta("add_fwd");
    addMeta.inputs_data.push_back(convMeta.outputs_data[0]);
    addMeta.inputs_data.push_back(expandMeta.outputs_data[0]);
    addMeta.outputs_data.emplace_back(inputRank, inputDtype);
    convolutionSharedMeta.push_back(addMeta);
  }

  if (is_conv_1d) {
    SharedMetaData squeezeMeta("squeeze");
    squeezeMeta.inputs_data.emplace_back(inputRank, inputDtype);
    squeezeMeta.outputs_data.emplace_back(--inputRank, inputDtype);
    convolutionSharedMeta.push_back(squeezeMeta);
  }

  return convolutionSharedMeta;
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

  // torch.Conv2D and torch.Conv1D always enter here with 4D or wider tensors
  // 3D case never happens as it could not be inferred if it is:
  // - torch.Conv1D (N,C,L) -> (N,C,L,1) or
  // - torch.Conv2D (C,H,W) -> (1,C,H,W)
  // 3D case can happen from torch.convolution
  IF_CONV1D_EXPAND_TO_2D(input, 0);
  IF_CONV1D_EXPAND_TO_2D(weight, 1);

  const bool is_conv_3d = input.dim() == 5;

  auto [in_layouts, out_layouts] = MakeLayouts(is_conv_3d, transposed);
  SetSynapseLayouts(in_layouts, out_layouts);

  std::string guid = transposed ? "dedx" : "spatial_convolution";
  if (is_conv_3d)
    guid += "3d";

  std::vector<synTensor> inputs = {input_expanded, weight_expanded};

  auto meta = ConvolutionOverrideableMeta(stack)[0];

  if (is_conv_1d)
    meta.shape.push_back(1);

  if (transposed)
    CreateShapeTensorInput(graph, meta.dtype, meta.shape, inputs);
  else if (bias.defined())
    inputs.emplace_back(syn_in(2));

  const auto& params = FillConvolutionOverrideableParams(stack, size);

  NodeAttr::NodeOutputAttr node_output_attr = {meta.shape, meta.dtype, 0};
  if ((transposed && bias.defined()) || is_conv_1d)
    node_output_attr.final_result_index = c10::nullopt;

  auto convOp = BuildOp(
      graph,
      std::move(guid),
      std::move(inputs),
      {node_output_attr},
      params.get(),
      size);

  SetSynapseLayouts({}, {});

  if (transposed && bias.defined()) {
    // Expand bias to match to NCHW output format
    int64_t data[5] = {1, bias.sizes().vec()[0], 1, 1, 1};
    c10::IntArrayRef shape(data, is_conv_3d ? 5 : 4);

    ns_ExpandMultiDimsKernel::Params expandParams;
    expandParams.expand_axes_mask = is_conv_3d ? 0b10111 : 0b1011;

    synapse_helpers::tensor biasExpanded = std::move(BuildOp(
        graph,
        "expand_multi_dims",
        {syn_in(2)},
        {{shape, meta.dtype}},
        &expandParams,
        sizeof(expandParams))[0]);

    c10::optional<int> final_result_index_0 =
        is_conv_1d ? c10::optional<int>{c10::nullopt} : c10::optional<int>{0};
    auto addOp = BuildOp(
        graph,
        get_guid_with_precision("add_fwd", meta.dtype),
        {convOp[0].get(), biasExpanded.get()},
        {{meta.shape, meta.dtype, final_result_index_0}});
    IF_CONV1D_SQUEEZE_TO_ORIG_AND_SET_OUT(addOp[0], meta.shape, 0);
  } else {
    IF_CONV1D_SQUEEZE_TO_ORIG_AND_SET_OUT(convOp[0], meta.shape, 0);
  }
}
} // namespace habana

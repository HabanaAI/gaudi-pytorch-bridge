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
#include "synapse_helpers/layout_utils.h"

using namespace torch;
using namespace habana;
using namespace synapse_helpers::layouts;

synConvolution3DParams synapse_conv3d_params_builder(
    const IntArrayRef& weight, // DHWCK
    const IntArrayRef& stride, // DHW
    const IntArrayRef& padding, // DHW
    const IntArrayRef& dilation, // DHW
    int64_t groups) {
  constexpr uint32_t d_axis = 0;
  constexpr uint32_t h_axis = 1;
  constexpr uint32_t w_axis = 2;
  const int64_t filter_D = weight[d_axis];
  const int64_t filter_H = weight[h_axis];
  const int64_t filter_W = weight[w_axis];
  const int64_t stride_D = stride[d_axis];
  const int64_t stride_H = stride[h_axis];
  const int64_t stride_W = stride[w_axis];
  const int64_t dilation_D = dilation[d_axis];
  const int64_t dilation_H = dilation[h_axis];
  const int64_t dilation_W = dilation[w_axis];

  synConvolution3DParams syn_conv_params{};
  syn_conv_params.kernel[CONV_KERNEL_WIDTH] = filter_W;
  syn_conv_params.kernel[CONV_KERNEL_HEIGHT] = filter_H;
  syn_conv_params.kernel[CONV_KERNEL_DEPTH] = filter_D;
  syn_conv_params.stride[CONV_STRIDE_WIDTH] = stride_W;
  syn_conv_params.stride[CONV_STRIDE_HEIGHT] = stride_H;
  syn_conv_params.stride[CONV_STRIDE_DEPTH] = stride_D;
  syn_conv_params.dilation[CONV_DIL_WIDTH] = dilation_W;
  syn_conv_params.dilation[CONV_DIL_HEIGHT] = dilation_H;
  syn_conv_params.dilation[CONV_DIL_DEPTH] = dilation_D;
  syn_conv_params.padding[CONV_PAD_LEFT] = padding[w_axis];
  syn_conv_params.padding[CONV_PAD_RIGHT] = padding[w_axis];
  syn_conv_params.padding[CONV_PAD_TOP] = padding[h_axis];
  syn_conv_params.padding[CONV_PAD_BOTTOM] = padding[h_axis];
  syn_conv_params.padding[CONV_PAD_FRONT] = padding[d_axis];
  syn_conv_params.padding[CONV_PAD_BACK] = padding[d_axis];
  syn_conv_params.nGroups = groups;

  return syn_conv_params;
}

synConvolutionParams synapse_conv_params_builder(
    const IntArrayRef& weight, // HWCK
    const IntArrayRef& stride, // HW
    const IntArrayRef& padding, // HW
    const IntArrayRef& dilation, // HW
    int64_t groups) {
  const int64_t filter_H = weight[WEIGHT_KERNEL_R_IDX];
  const int64_t filter_W = weight[WEIGHT_KERNEL_S_IDX];
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

bool is_5d_tensor(
    const std::vector<int64_t>& shape_in,
    const std::vector<int64_t>& shape_wt) {
  TORCH_CHECK(
      shape_in.size() == shape_wt.size(),
      "Inputs and weights dimensions do not match");
  const uint64_t DIM5 = 5;
  return shape_in.size() == DIM5;
}

bool is_5d_tensor(const std::vector<at::Tensor>& inputs) {
  TORCH_CHECK(
      inputs.size() >= 2, "Need to check both inputs and weights for 5D");
  TORCH_CHECK(
      inputs[0].dim() == inputs[1].dim(),
      "Inputs and weights dimensions do not match");
  return is_5d_tensor(inputs[0].sizes().vec(), inputs[1].sizes().vec());
}

std::vector<int64_t> ConvOperator::compute_output_shape(
    std::vector<int64_t> shape_in,
    std::vector<int64_t> shape_wt,
    std::vector<int64_t> pad,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation,
    const bool ceil_mode,
    const bool transposed) {
  TORCH_CHECK(ceil_mode == false, "No support for ceil_mode");
  TORCH_CHECK(
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING),
      "compute_output_shape only for Synapse layout handling mode");

  bool conv3d = is_5d_tensor(shape_in, shape_wt);
  return conv3d
      ? compute_output_shape_3d(
            shape_in, shape_wt, pad, stride, dilation, ceil_mode, transposed)
      : compute_output_shape_2d(
            shape_in, shape_wt, pad, stride, dilation, ceil_mode, transposed);
}

std::vector<int64_t> ConvOperator::compute_output_shape_2d(
    std::vector<int64_t> shape_in,
    std::vector<int64_t> shape_wt,
    std::vector<int64_t> pad,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation,
    const bool ceil_mode,
    const bool transposed) {
  TORCH_CHECK(ceil_mode == false, "No support for ceil_mode");
  TORCH_CHECK(
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING),
      "compute_output_shape only for Synapse layout handling mode");
  TORCH_CHECK(
      !is_5d_tensor(shape_in, shape_wt),
      "compute_output_shape of 2d conv kernels");

  const auto output_H = compute_output_single_dim(
      shape_in,
      shape_wt,
      pad,
      stride,
      dilation,
      INPUT_H_IDX,
      WEIGHT_KERNEL_R_IDX,
      CONV2D_KERNEL_HIEGHT_ATTRIBUTE_IDX,
      transposed);
  const auto output_W = compute_output_single_dim(
      shape_in,
      shape_wt,
      pad,
      stride,
      dilation,
      INPUT_W_IDX,
      WEIGHT_KERNEL_S_IDX,
      CONV2D_KERNEL_WIDTH_ATTRIBUTE_IDX,
      transposed);

  auto K = transposed ? shape_wt[1] : shape_wt[WEIGHT_KERNEL_K_IDX];

  std::vector<int64_t> out_shape{shape_in[INPUT_N_IDX], K, output_H, output_W};
  return out_shape;
}

std::vector<int64_t> ConvOperator::compute_output_shape_3d(
    std::vector<int64_t> shape_in,
    std::vector<int64_t> shape_wt,
    std::vector<int64_t> pad,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation,
    const bool ceil_mode,
    const bool transposed) {
  TORCH_CHECK(ceil_mode == false, "No support for ceil_mode");
  TORCH_CHECK(
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING),
      "compute_output_shape only for Synapse layout handling mode");
  TORCH_CHECK(
      is_5d_tensor(shape_in, shape_wt),
      "compute_output_shape of 3d conv kernels");

  const auto output_D = compute_output_single_dim(
      shape_in,
      shape_wt,
      pad,
      stride,
      dilation,
      INPUT_3D_D_IDX,
      WEIGHT_KERNEL_3D_Q_IDX,
      CONV3D_KERNEL_DEPTH_ATTRIBUTE_IDX,
      transposed);
  const auto output_H = compute_output_single_dim(
      shape_in,
      shape_wt,
      pad,
      stride,
      dilation,
      INPUT_3D_H_IDX,
      WEIGHT_KERNEL_3D_R_IDX,
      CONV3D_KERNEL_HIEGHT_ATTRIBUTE_IDX,
      transposed);
  const auto output_W = compute_output_single_dim(
      shape_in,
      shape_wt,
      pad,
      stride,
      dilation,
      INPUT_3D_W_IDX,
      WEIGHT_KERNEL_3D_S_IDX,
      CONV3D_KERNEL_WIDTH_ATTRIBUTE_IDX,
      transposed);

  auto K = transposed ? shape_wt[1] : shape_wt[WEIGHT_KERNEL_3D_K_IDX];

  std::vector<int64_t> out_shape{
      shape_in[INPUT_3D_N_IDX], K, output_D, output_H, output_W};
  return out_shape;
}

int64_t ConvOperator::compute_output_single_dim(
    std::vector<int64_t> shape_in,
    std::vector<int64_t> shape_wt,
    std::vector<int64_t> padding,
    std::vector<int64_t> strides,
    std::vector<int64_t> dilation,
    unsigned input_idx,
    unsigned kernel_idx,
    unsigned attributes_idx,
    bool transposed) {
  const auto input = shape_in[input_idx];
  const auto pad = padding[attributes_idx];
  const auto dil = dilation[attributes_idx];
  const auto filter = shape_wt[kernel_idx];
  const auto stride = strides[attributes_idx];
  return habana_helpers::compute_output_size(
      input, pad, dil, filter, stride, false, transposed);
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
    std::vector<int64_t> dilation,
    const bool ceil_mode,
    const bool transposed,
    c10::MemoryFormat memory_format,
    const bool is_conv_3d,
    const bool is_weight_hwck) {
  TORCH_CHECK(ceil_mode == false, "No support for ceil_mode");
  TORCH_CHECK(
      (memory_format == c10::MemoryFormat::ChannelsLast3d) ||
          (memory_format == c10::MemoryFormat::ChannelsLast) ||
          (memory_format == c10::MemoryFormat::Contiguous),
      "Only ChannelsLast3d, ChannelsLast and Contiguous supported");

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    return compute_output_shape(
        shape_in, shape_wt, pad, stride, dilation, ceil_mode, transposed);
  }

  // the following is required to this function being called during
  // both the lazy and the eager modes of execution with different
  // entry points
  auto is_conv_3d_recheck = is_conv_3d || is_5d_tensor(shape_in, shape_wt);
  if (is_conv_3d_recheck) {
    const int64_t dim_pos_in[5] = {0, 2, 3, 4, 1};
    const int64_t dim_pos_wt[5] = {0, 1, 2, 3, 4};
    const int64_t dim_pos_in_chlast[5] = {0, 1, 2, 3, 4};
    const int64_t wt_hwck_dims[5] = {2, 3, 4, 1, 0};

    const int64_t* p_dim_pos_in;
    const int64_t* p_dim_pos_wt;

    TORCH_CHECK(
        memory_format != c10::MemoryFormat::ChannelsLast,
        "Memory format should be ChannelsLast3d/Contiguous in Conv3d");

    if (memory_format == c10::MemoryFormat::ChannelsLast3d) {
      p_dim_pos_in = dim_pos_in_chlast;
    } else {
      p_dim_pos_in = dim_pos_in;
    }

    p_dim_pos_wt = dim_pos_wt;
    if (!is_weight_hwck) {
      p_dim_pos_wt = wt_hwck_dims;
    }
    const auto input_D = shape_in[p_dim_pos_in[1]];
    const auto pad_D = pad[0];
    const auto dil_D = dilation[0];
    const auto filter_D = shape_wt[p_dim_pos_wt[0]];
    const auto stride_D = stride[0];

    const auto output_D = habana_helpers::compute_output_size(
        input_D, pad_D, dil_D, filter_D, stride_D, false, transposed);

    const auto input_H = shape_in[p_dim_pos_in[2]];
    const auto pad_H = pad[1];
    const auto dil_H = dilation[1];
    const auto filter_H = shape_wt[p_dim_pos_wt[1]];
    const auto stride_H = stride[1];

    const auto output_H = habana_helpers::compute_output_size(
        input_H, pad_H, dil_H, filter_H, stride_H, false, transposed);

    const auto input_W = shape_in[p_dim_pos_in[3]];
    const auto pad_W = pad[2];
    const auto dil_W = dilation[2];
    const auto filter_W = shape_wt[p_dim_pos_wt[2]];
    const auto stride_W = stride[2];

    const auto output_W = habana_helpers::compute_output_size(
        input_W, pad_W, dil_W, filter_W, stride_W, false, transposed);

    auto K = shape_wt[p_dim_pos_wt[4]];

    if (transposed) {
      // for conv_transpose3d weights are in DHWKC format
      K = shape_wt[p_dim_pos_wt[3]];
    }
    std::vector<int64_t> out_shape;

    if (memory_format == c10::MemoryFormat::ChannelsLast3d) {
      out_shape.push_back(shape_in[0]);
      out_shape.push_back(output_D);
      out_shape.push_back(output_H);
      out_shape.push_back(output_W);
      out_shape.push_back(K);
    } else {
      out_shape.push_back(shape_in[0]);
      out_shape.push_back(K);
      out_shape.push_back(output_D);
      out_shape.push_back(output_H);
      out_shape.push_back(output_W);
    }
    return out_shape;
  } else {
    const int64_t dim_pos_in[4] = {0, 2, 3, 1};
    const int64_t dim_pos_wt[4] = {0, 1, 2, 3};
    const int64_t dim_pos_in_chlast[4] = {0, 1, 2, 3};
    const int64_t wt_hwck_dims[4] = {2, 3, 1, 0};
    const int64_t* p_dim_pos_in;
    const int64_t* p_dim_pos_wt;

    TORCH_CHECK(
        memory_format != c10::MemoryFormat::ChannelsLast3d,
        "Memory format should be ChannelsLast/Contiguous in Conv2d");
    if (memory_format == c10::MemoryFormat::ChannelsLast) {
      p_dim_pos_in = dim_pos_in_chlast;
    } else {
      p_dim_pos_in = dim_pos_in;
    }

    p_dim_pos_wt = dim_pos_wt;
    if (!is_weight_hwck) {
      p_dim_pos_wt = wt_hwck_dims;
    }

    const auto input_H = shape_in[p_dim_pos_in[1]];
    const auto pad_H = pad[0];
    const auto dil_H = dilation[0];
    const auto filter_H = shape_wt[p_dim_pos_wt[0]];
    const auto stride_H = stride[0];

    const auto output_H = habana_helpers::compute_output_size(
        input_H, pad_H, dil_H, filter_H, stride_H, false, transposed);

    const auto input_W = shape_in[p_dim_pos_in[2]];
    const auto pad_W = pad[1];
    const auto dil_W = dilation[1];
    const auto filter_W = shape_wt[p_dim_pos_wt[1]];
    const auto stride_W = stride[1];
    const auto output_W = habana_helpers::compute_output_size(
        input_W, pad_W, dil_W, filter_W, stride_W, false, transposed);

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
}

void SpatialConv3DOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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

  auto weight_channel = WEIGHT_KERNEL_3D_C_IDX;
  if (transposed) {
    // conv_transpose3d realized using conv_backward w.r.t input, so use guid
    // corresponding to that
    std::string id = "dedx3d";
    SetGuid(id);
    // conv_transpose2d weights are in DHWKC format
    weight_channel =
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) ? 0 : 4;
  }

  habana_helpers::check_convolution_params(
      pt_inputs,
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(dilation),
      transposed,
      IntArrayRef(output_padding),
      groups,
      INPUT_3D_C_IDX, /*input_channel*/
      weight_channel,
      true /*is_conv_3d*/);

  // input, output NCDHW
  // weight KCDHW, where K - output channels
  // pad, stride DHW
  c10::MemoryFormat memory_format =
      habana_helpers::get_memory_format({&input, &weight});

  // permute happened outside this function. Hence always set channelsLast
  // format
  std::vector<int64_t> shape_out = ConvOperator::compute_output_shape(
      input.sizes().vec(),
      weight.sizes().vec(),
      padding,
      stride,
      dilation,
      false,
      transposed,
      c10::MemoryFormat::ChannelsLast3d,
      true /*is_conv_3d*/);

  auto output = habana_helpers::createPTTensor(
      input,
      shape_out,
      input.options(),
      memory_format,
      output_metadata.at(0).persistent);

  // Allocate Shape Tensor (only for conv_tranpose2d which uses dedx node)
  if (graph.is_dynamic_graph() && this->guid_ == "dedx3d") {
    AllocateSynapseShapeTensor(graph, output);
  }

  synConvolution3DParams params = synapse_conv3d_params_builder(
      weight.sizes(),
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(dilation),
      groups);

  p_context_->params_.emplace<synConvolution3DParams>(params);
  p_context_->params_size_ = sizeof(params);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void SpatialConvOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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

  auto weight_channel = WEIGHT_KERNEL_C_IDX;
  if (transposed) {
    // conv_transpose2d realized using conv_backward w.r.t input, so use guid
    // corresponding to that
    SetGuid("dedx");
    // conv_transpose2d weights are in HWKC format
    weight_channel =
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) ? 0 : 3;
  }

  habana_helpers::check_convolution_params(
      pt_inputs,
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(dilation),
      transposed,
      IntArrayRef(output_padding),
      groups,
      INPUT_C_IDX, /*input_channel*/
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
      dilation,
      false,
      transposed,
      c10::MemoryFormat::ChannelsLast);

  auto output = habana_helpers::createPTTensor(
      input,
      shape_out,
      input.options(),
      memory_format,
      output_metadata.at(0).persistent);

  // Allocate Shape Tensor (only for conv_tranpose2d which uses dedx node)
  if (graph.is_dynamic_graph() && this->guid_ == "dedx") {
    AllocateSynapseShapeTensor(graph, output);
  }

  synConvolutionParams params = synapse_conv_params_builder(
      weight.sizes(),
      IntArrayRef(stride),
      IntArrayRef(padding),
      IntArrayRef(dilation),
      groups);

  p_context_->params_.emplace<synConvolutionParams>(params);
  p_context_->params_size_ = sizeof(params);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void ConvOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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
  at::Tensor weight = inputs[1].toTensor();
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

  std::vector<at::Tensor> pt_inputs{input, weight};
  auto is_conv_3d = is_5d_tensor(pt_inputs);

  if (!bias.defined()) {
    auto populateOp =
        [&](std::shared_ptr<habana::HabanaOperator> scOp) mutable {
          scOp->SetSynapseInput(p_context_->syn_inputs_[0]);
          scOp->SetSynapseInput(p_context_->syn_inputs_[1]);
          scOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);

          p_context_->syn_outputs_.emplace_back(
              std::move(scOp->GetSynOutputs()[0]));
          p_context_->pt_outputs_.emplace_back(
              std::move(scOp->GetOutputs()[0]));
        };
    if (is_conv_3d) {
      auto scOp = make_operator<SpatialConv3DOperator>(
          this->p_context_->device_id_, input.scalar_type());
      populateOp(scOp);
    } else {
      auto scOp = make_operator<SpatialConvOperator>(
          this->p_context_->device_id_, input.scalar_type());
      populateOp(scOp);
    }
  } else {
    if (!transposed) {
      auto populateOp =
          [&](std::shared_ptr<habana::HabanaOperator> scOp) mutable {
            scOp->SetSynapseInput(p_context_->syn_inputs_[0]);
            scOp->SetSynapseInput(p_context_->syn_inputs_[1]);
            scOp->SetSynapseInput(p_context_->syn_inputs_[2]);
            scOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);

            p_context_->syn_outputs_.emplace_back(
                std::move(scOp->GetSynOutputs()[0]));
            p_context_->pt_outputs_.emplace_back(
                std::move(scOp->GetOutputs()[0]));
          };
      if (is_conv_3d) {
        auto scOp = make_operator<SpatialConv3DOperator>(
            this->p_context_->device_id_, input.scalar_type());
        populateOp(scOp);
      } else {
        auto scOp = make_operator<SpatialConvOperator>(
            this->p_context_->device_id_, input.scalar_type());
        populateOp(scOp);
      }
    } else {
      auto populateOp =
          [&](std::shared_ptr<habana::HabanaOperator> scOp) mutable {
            scOp->SetSynapseInput(p_context_->syn_inputs_[0]);
            scOp->SetSynapseInput(p_context_->syn_inputs_[1]);
            scOp->AllocateAndAddSynapseNode(
                graph, inputs, OutputMetaDataVector(1));

            auto addOp = make_operator<AddOperator>(
                this->p_context_->device_id_, input.scalar_type());
            addOp->SetSynapseInput(scOp->GetSynOutputs()[0]);
            addOp->SetSynapseInput(p_context_->syn_inputs_[2]);
            // Build Params for the graph
            Scalar alphaValue = 1.0;
            torch::jit::Stack stack;
            stack.emplace_back(IValue(scOp->GetOutputs()[0]));
            stack.emplace_back(IValue(bias));
            stack.emplace_back(IValue(alphaValue));
            addOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
            stack.clear();

            p_context_->syn_outputs_.emplace_back(
                std::move(addOp->GetSynOutputs()[0]));
            p_context_->pt_outputs_.emplace_back(
                std::move(addOp->GetOutputs()[0]));
          };
      if (is_conv_3d) {
        auto scOp = make_operator<SpatialConv3DOperator>(
            this->p_context_->device_id_, input.scalar_type());
        populateOp(scOp);
      } else {
        auto scOp = make_operator<SpatialConvOperator>(
            this->p_context_->device_id_, input.scalar_type());
        populateOp(scOp);
      }
    }
  }
}

void ConvOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  at::Tensor input = inputs[0].toTensor();
  at::Tensor weight = inputs[1].toTensor();
  const auto stride = inputs[3].toIntList().vec();
  const auto padding = inputs[4].toIntList().vec();
  const auto dilation = inputs[5].toIntList().vec();
  const bool transposed = inputs[6].toBool();

  c10::MemoryFormat memory_format =
      habana_helpers::get_memory_format({&input, &weight});

  std::vector<at::Tensor> pt_inputs{input, weight};
  auto is_conv_3d = is_5d_tensor(pt_inputs);
  auto format = is_conv_3d ? c10::MemoryFormat::ChannelsLast3d
                           : c10::MemoryFormat::ChannelsLast;

  auto shape_out = compute_output_shape(
      input.sizes().vec(),
      weight.sizes().vec(),
      padding,
      stride,
      dilation,
      false,
      transposed,
      format,
      is_conv_3d);

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
  auto is_conv_3d = is_5d_tensor(inputs);
  int64_t pos_in[4] = {0, 2, 3, 1};
  int64_t pos_w[4] = {2, 3, 1, 0};
  int64_t pos_in_3d[5] = {0, 2, 3, 4, 1};
  int64_t pos_w_3d[5] = {2, 3, 4, 1, 0};

  std::vector<const at::Tensor*> pt_in{&input};
  std::vector<at::Tensor*> pt_out{&input_nhwc};
  IntArrayRef new_dim_pos_in = pos_in;
  IntArrayRef new_dim_pos_w = pos_w;
  if (is_conv_3d) {
    new_dim_pos_in = pos_in_3d;
    new_dim_pos_w = pos_w_3d;
  }
  std::vector<const IntArrayRef*> pt_new_pos{&new_dim_pos_in, &new_dim_pos_w};
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  // habana_helpers::change_tensor_strides(&weight_hwck, &weight,
  // &new_dim_pos_w);

  auto convolution = [&] {
    size_t device_id = input.device().index();
    at::ScalarType scalar_type = input.scalar_type();
    std::string node_type =
        is_conv_3d ? "spatial_convolution3d" : "spatial_convolution";
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
      OutputMetaDataVector output_metadata(1);
      output_metadata.at(0).persistent = true;
      Op.AllocateAndAddSynapseNode(graph, stack, output_metadata);
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
  int64_t pos_out[4] = {0, 3, 1, 2};
  int64_t pos_out_3d[5] = {0, 4, 1, 2, 3};

  IntArrayRef new_dim_pos_out = pos_out;
  if (is_conv_3d) {
    new_dim_pos_out = pos_out_3d;
  }
  pt_new_pos = {&new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  PT_KERNEL_END;
  return output;
}

static auto& KernelRegistry = habana::KernelRegistry().add(
    "aten::convolution_overrideable",
    KERNEL_FN(ConvOperator));

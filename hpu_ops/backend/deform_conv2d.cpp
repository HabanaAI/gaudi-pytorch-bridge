/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/deform_conv2d.h"

namespace habana {

OutputMetaDataVector DeformConv2dOutputMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);
  const auto& weight = stack_tensor(stack, 1);
  const auto& offset = stack_tensor(stack, 2);
  const auto& mask = stack_tensor(stack, 3);
  const auto stride_h = stack[5].toInt();
  const auto stride_w = stack[6].toInt();
  const auto pad_h = stack[7].toInt();
  const auto pad_w = stack[8].toInt();
  const auto dilation_h = stack[9].toInt();
  const auto dilation_w = stack[10].toInt();
  const auto n_weight_grps = stack[11].toInt();
  const auto n_offset_grps = stack[12].toInt();
  const auto use_mask = stack[13].toBool();

  const int batch_sz = input.size(0);
  const int in_h = input.size(2);
  const int in_w = input.size(3);

  const int out_channels = weight.size(0);
  const int weight_h = weight.size(2);
  const int weight_w = weight.size(3);

  int ker_h = dilation_h * (weight_h - 1) + 1;
  int ker_w = dilation_w * (weight_w - 1) + 1;
  int out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
  int out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;

  OutputMetaData meta;
  meta.shape = {batch_sz, out_channels, out_h, out_w};
  meta.dtype = input.scalar_type();
  meta.mem_format = input.suggest_memory_format();

  TORCH_CHECK(input.ndimension() == 4);
  TORCH_CHECK(offset.ndimension() == 4);
  TORCH_CHECK(!use_mask || mask.ndimension() == 4);
  TORCH_CHECK(weight.ndimension() == 4);

  TORCH_CHECK(
      weight_h > 0 && weight_w > 0,
      "weight_h: ",
      weight_h,
      " weight_w: ",
      weight_w);
  TORCH_CHECK(
      stride_h > 0 && stride_w > 0,
      "stride_h: ",
      stride_h,
      " stride_w: ",
      stride_w);
  TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "pad_h: ", pad_h, " pad_w: ", pad_w);
  TORCH_CHECK(
      dilation_h > 0 && dilation_w > 0,
      "dilation_h: ",
      dilation_h,
      " dilation_w: ",
      dilation_w);

  TORCH_CHECK(weight.size(1) * n_weight_grps == input.size(1));
  TORCH_CHECK(weight.size(0) % n_weight_grps == 0);
  TORCH_CHECK(
      (offset.size(1) == n_offset_grps * 2 * weight_h * weight_w),
      "offset.shape[1] is not valid: got: ",
      offset.size(1),
      " expected: ",
      n_offset_grps * 2 * weight_h * weight_w);
  TORCH_CHECK(
      (!use_mask || mask.size(1) == n_offset_grps * weight_h * weight_w),
      "mask.shape[1] is not valid: got: ",
      mask.size(1),
      " expected: ",
      n_offset_grps * weight_h * weight_w);
  TORCH_CHECK(input.size(1) % n_offset_grps == 0);

  TORCH_CHECK(
      (offset.size(0) == input.size(0)), "invalid batch size of offset");
  TORCH_CHECK(
      (offset.size(2) == out_h && offset.size(3) == out_w),
      "offset output dims: (",
      offset.size(2),
      ", ",
      offset.size(3),
      ") - ",
      "computed output dims: (",
      out_h,
      ", ",
      out_w,
      ")");
  TORCH_CHECK((mask.size(0) == input.size(0)), "invalid batch size of mask");
  TORCH_CHECK(
      (!use_mask || (mask.size(2) == out_h && mask.size(3) == out_w)),
      "mask output dims: (",
      mask.size(2),
      ", ",
      mask.size(3),
      ") - ",
      "computed output dims: (",
      out_h,
      ", ",
      out_w,
      ")");
  TORCH_CHECK(
      out_h > 0 && out_w > 0,
      "Calculated output size too small - out_h: ",
      out_h,
      " out_w: ",
      out_w);

  return {meta};
}

DeformConv2d::DeformConv2d(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "deform_conv2d_fwd",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetOutputMetaFn(DeformConv2dOutputMeta);
  SetSynapseLayouts(
      {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::SRCK,
       synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE},
      {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
}

void DeformConv2d::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);

  std::vector<synTensor> syn_inputs{
      syn_in(0), syn_in(2), syn_in(1), syn_in(3), syn_in(4)};
  ns_DeformConv::Params params{};
  params.strideW = stack[6].toInt();
  params.strideH = stack[5].toInt();
  params.padW = stack[8].toInt();
  params.padH = stack[7].toInt();
  params.dilationW = stack[10].toInt();
  params.dilationH = stack[9].toInt();

  auto meta = DeformConv2dOutputMeta(stack)[0];
  syn_out(0) = std::move(BuildOp(
      graph,
      get_guid_with_precision("deform_conv", input.scalar_type()),
      std::move(syn_inputs),
      {{meta.shape, meta.dtype, 0}},
      &params,
      sizeof(params))[0]);
}

OutputMetaDataVector DeformConv2dBackwardOutputMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 1);
  const auto& weight = stack_tensor(stack, 2);
  const auto& offset = stack_tensor(stack, 3);
  const auto& mask = stack_tensor(stack, 4);
  const auto& bias = stack_tensor(stack, 5);

  OutputMetaData meta_input{input.scalar_type(), input.sizes().vec()};
  OutputMetaData meta_weight{weight.scalar_type(), weight.sizes().vec()};
  OutputMetaData meta_offset{offset.scalar_type(), offset.sizes().vec()};
  OutputMetaData meta_mask{mask.scalar_type(), mask.sizes().vec()};
  OutputMetaData meta_bias{bias.scalar_type(), bias.sizes().vec()};

  return {meta_input, meta_weight, meta_offset, meta_mask, meta_bias};
}

DeformConv2dBackward::DeformConv2dBackward(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "deform_conv2d_bwd",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetOutputMetaFn(DeformConv2dBackwardOutputMeta);
  SetSynapseLayouts(
      {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::SRCK,
       synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
      {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::SRCK,
       synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
}

void DeformConv2dBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 1);

  std::vector<synTensor> syn_inputs{
      syn_in(0), syn_in(1), syn_in(2), syn_in(3), syn_in(4)};

  auto meta = DeformConv2dBackwardOutputMeta(stack);
  auto grads = BuildOp(
      graph,
      get_guid_with_precision("deform_conv_bwd", input.scalar_type()),
      std::move(syn_inputs),
      {{meta[0].shape, meta[0].dtype, 0},
       {meta[1].shape, meta[1].dtype, 1},
       {meta[2].shape, meta[2].dtype, 2},
       {meta[3].shape, meta[3].dtype, 3},
       {meta[4].shape, meta[4].dtype, 4}});

  syn_out(0) = std::move(grads[0]);
  syn_out(1) = std::move(grads[1]);
  syn_out(2) = std::move(grads[2]);
  syn_out(3) = std::move(grads[3]);
  syn_out(4) = std::move(grads[4]);
}

} // namespace habana

static const auto& DeformConv2dKernelRegistry =
    habana::KernelRegistry()
        .add(
            "torchvision::deform_conv2d",
            KERNEL_FN_GLOBAL(habana::DeformConv2d))
        .add(
            "torchvision::_deform_conv2d_backward",
            KERNEL_FN_GLOBAL(habana::DeformConv2dBackward));

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
  const auto stride_h = stack[5].toInt();
  const auto stride_w = stack[6].toInt();
  const auto pad_h = stack[7].toInt();
  const auto pad_w = stack[8].toInt();
  const auto dilation_h = stack[9].toInt();
  const auto dilation_w = stack[10].toInt();

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
       synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
      {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
}

void DeformConv2d::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);

  std::vector<synTensor> syn_inputs{syn_in(0), syn_in(2), syn_in(1), syn_in(3)};
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
       synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
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
       {meta[3].shape, meta[3].dtype, 3}});
  syn_out(0) = std::move(grads[0]);
  syn_out(1) = std::move(grads[1]);
  syn_out(2) = std::move(grads[2]);
  syn_out(3) = std::move(grads[3]);
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

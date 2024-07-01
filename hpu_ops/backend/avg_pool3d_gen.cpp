/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/avg_pool3d.h"
#include "generated/backend/avg_pool3d_backward.h"
#include "hpu_ops/backend/pool_helpers.h"
#include "hpu_ops/shared_meta_common.h"

#define CHECK_DIM(input_size)                                             \
  TORCH_CHECK(                                                            \
      input_size == 4 || input_size == 5,                                 \
      "Averagepool3D expects input_size equals to 4 or 5, but got size ", \
      input_size);

namespace habana {
static int64_t GetParam(const std::vector<int64_t>& params, size_t position) {
  return params.at(params.size() == 1 ? 0 : position);
}

static std::shared_ptr<void> FillAvgPool3dParams(
    const std::vector<int64_t>& kernelSize,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& pad,
    bool ceilMode,
    bool includePadding,
    int64_t divOverride,
    size_t& size) {
  PARAMS_STUB(ns_AveragePooling3DWithDivisorOverride::Params);
  params->pad_w_begin = GetParam(pad, 2);
  params->pad_w_end = GetParam(pad, 2);
  params->pad_h_begin = GetParam(pad, 1);
  params->pad_h_end = GetParam(pad, 1);
  params->pad_d_begin = pad.at(0);
  params->pad_d_end = pad.at(0);
  params->kernel_w = GetParam(kernelSize, 2);
  params->kernel_h = GetParam(kernelSize, 1);
  params->kernel_d = kernelSize.at(0);
  params->stride_w = GetParam(stride, 2);
  params->stride_h = GetParam(stride, 1);
  params->stride_d = stride.at(0);
  params->dilation_w = 1;
  params->dilation_h = 1;
  params->dilation_d = 1;
  params->includePadding = includePadding ? 1 : 0;
  params->divisorOverride = divOverride;
  params->pooling_convention = ceilMode
      ? EPoolingConvention::POOLING_CONVENTION_FULL_PYTORCH
      : EPoolingConvention::POOLING_CONVENTION_VALID;
  return params;
}

std::shared_ptr<void> FillAvgPool3dParamsFwd(
    const at::Stack& stack,
    size_t& size) {
  std::vector<int64_t> defaultPadding = {0};
  auto kernelSize = stack.at(1).toIntVector();
  auto stride =
      stack.at(2).toListRef().empty() ? kernelSize : stack.at(2).toIntVector();
  auto padding =
      stack.at(3).isNone() ? defaultPadding : stack.at(3).toIntVector();
  const bool ceilMode = stack.at(4).toBool();
  const bool includePadding = stack.at(5).toBool();
  int64_t divOverride = stack.at(6).isNone() ? 0 : stack.at(6).toInt();
  return FillAvgPool3dParams(
      kernelSize, stride, padding, ceilMode, includePadding, divOverride, size);
}

OutputMetaDataVector AvgPool3dMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  const int rank = self.dim();
  CHECK_DIM(rank);

  std::vector<int64_t> defaultPadding = {0, 0, 0};
  std::vector<int64_t> dilation = {1, 1, 1};
  auto kernelSize = stack.at(1).toIntVector();
  auto stride =
      stack.at(2).toListRef().empty() ? kernelSize : stack.at(2).toIntVector();
  auto padding =
      stack.at(3).isNone() ? defaultPadding : stack.at(3).toIntVector();
  const bool ceilMode = stack.at(4).toBool();
  auto outshape = compute_pool_kernel_output_shape(
      self, kernelSize, stride, padding, dilation, ceilMode, true);
  if (rank == 4)
    outshape.erase(begin(outshape));

  OutputMetaData meta;
  meta.shape = outshape;
  meta.dtype = self.scalar_type();
  return {meta};
}

SharedMetaDataVector AvgPool3dFwdSharedMeta(const at::Stack& stack) {
  return Input0SharedMeta(stack, "avg_pool_3d_fwd");
}

void Avgpool3dFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = FillAvgPool3dParamsFwd(stack, size);
  auto meta = AvgPool3dMeta(stack)[0];
  const auto rank = stack_tensor(stack, 0).dim();
  if (rank == 4) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDC,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDC},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDC});
  } else if (rank == 5) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN});
  }
  auto avgPool = BuildOp(
      graph,
      guid_,
      {syn_in(0)},
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(avgPool[0]);
}

std::shared_ptr<void> FillAvgPool3dParamsBwd(
    const at::Stack& stack,
    size_t& size) {
  std::vector<long int> padding = {0, 0, 0};
  auto kernelSize = stack.at(2).toIntVector();
  auto stride =
      stack.at(3).toListRef().empty() ? kernelSize : stack.at(3).toIntVector();
  auto pad = stack.at(4).isNone() ? padding : stack.at(4).toIntVector();
  const bool ceilMode = stack.at(5).toBool();
  const bool includePad = stack.at(6).toBool();
  int64_t divOverride = stack.at(7).isNone() ? 0 : stack.at(7).toInt();
  return FillAvgPool3dParams(
      kernelSize, stride, pad, ceilMode, includePad, divOverride, size);
}

OutputMetaDataVector AvgPool3dBwdMeta(const at::Stack& stack) {
  auto self = stack.at(1).toTensor();
  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  return {meta};
}

SharedMetaDataVector AvgPool3dBwdSharedMeta(const at::Stack& stack) {
  return AvgPoolBwdSharedMeta(stack, "avg_pool_3d_bwd");
}

void AvgPool3dBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  auto meta = OutputMeta(stack)[0];
  const auto rank = stack_tensor(stack, 1).dim();
  if (rank == 4) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDC,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDC},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDC});
  } else if (rank == 5) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN});
  }

  std::vector<synTensor> inputs = {syn_in(0)};
  CreateShapeTensorInput(graph, meta.dtype, meta.shape, inputs);
  auto avgPool = BuildOp(
      graph,
      GetGuid(),
      std::move(inputs),
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(avgPool[0]);
}

} // namespace habana
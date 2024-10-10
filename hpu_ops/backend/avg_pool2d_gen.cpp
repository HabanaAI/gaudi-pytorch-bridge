/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/avg_pool2d.h"
#include "generated/backend/avg_pool2d_backward.h"
#include "hpu_ops/backend/pool_helpers.h"
#include "hpu_ops/shared_meta_common.h"

#define CHECK_DIM(input_size)                                             \
  TORCH_CHECK(                                                            \
      input_size == 3 || input_size == 4,                                 \
      "Averagepool2D expects input_size equals to 3 or 4, but got size ", \
      input_size);

namespace habana {

static std::shared_ptr<void> FillAvgpool2dParams(
    std::vector<int64_t>& kernel_size,
    std::vector<int64_t>& stride,
    std::vector<int64_t>& pad,
    bool ceil_mode,
    bool include_pad,
    int64_t divOverride,
    size_t& size) {
  PARAMS_STUB(ns_AveragePoolingWithDivisorOverride::Params);
  params->pad_w_begin = pad.size() == 1 ? pad.at(0) : pad.at(1);
  params->pad_w_end = pad.size() == 1 ? pad.at(0) : pad.at(1);
  params->pad_h_begin = pad.at(0);
  params->pad_h_end = pad.at(0);
  params->kernel_w =
      kernel_size.size() == 1 ? kernel_size.at(0) : kernel_size.at(1);
  params->kernel_h = kernel_size.at(0);
  params->stride_w = stride.size() == 1 ? stride.at(0) : stride.at(1);
  params->stride_h = stride.at(0);
  params->dilation_w = 1; // Dilation set to 1, since for AvgPool Pytorch API
                          // does not give dilation value
  params->dilation_h = 1;
  params->includePadding = include_pad ? 1 : 0;
  params->divisorOverride = divOverride;
  params->pooling_convention = ceil_mode
      ? EPoolingConvention::POOLING_CONVENTION_FULL_PYTORCH
      : EPoolingConvention::POOLING_CONVENTION_VALID;
  return params;
}
std::shared_ptr<void> Fillavgpool2dParamsFwd(
    const at::Stack& stack,
    size_t& size) {
  std::vector<long int> padding = {0, 0};
  auto kernel_size = stack.at(1).toIntVector();
  auto stride =
      stack.at(2).toListRef().empty() ? kernel_size : stack.at(2).toIntVector();
  auto pad =
      stack.at(3).toListRef().empty() ? padding : stack.at(3).toIntVector();
  const bool ceil_mode = stack.at(4).toBool();
  const bool include_pad = stack.at(5).toBool();
  int64_t divOverride = stack.at(6).isNone() ? 0 : stack.at(6).toInt();
  return FillAvgpool2dParams(
      kernel_size, stride, pad, ceil_mode, include_pad, divOverride, size);
}
std::shared_ptr<void> Fillavgpool2dParamsBwd(
    const at::Stack& stack,
    size_t& size) {
  std::vector<long int> padding = {0, 0};
  auto kernel_size = stack.at(2).toIntVector();
  auto stride =
      stack.at(3).toListRef().empty() ? kernel_size : stack.at(3).toIntVector();
  auto pad =
      stack.at(4).toListRef().empty() ? padding : stack.at(4).toIntVector();
  const bool ceil_mode = stack.at(5).toBool();
  const bool include_pad = stack.at(6).toBool();
  int64_t divOverride = stack.at(7).isNone() ? 0 : stack.at(7).toInt();
  return FillAvgpool2dParams(
      kernel_size, stride, pad, ceil_mode, include_pad, divOverride, size);
}

OutputMetaDataVector Avgpool2dMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const int rank = self.dim();
  CHECK_DIM(rank);
  std::vector<long int> padding = {0, 0};
  std::vector<int64_t> dilation = {1, 1};
  auto kernel_size = stack.at(1).toIntVector();
  auto stride =
      stack.at(2).toListRef().empty() ? kernel_size : stack.at(2).toIntVector();
  auto pad =
      stack.at(3).toListRef().empty() ? padding : stack.at(3).toIntVector();
  const bool ceil_mode = stack.at(4).toBool();
  auto outshape = compute_pool_kernel_output_shape(
      self, kernel_size, stride, pad, dilation, ceil_mode, false);
  if (rank == 3)
    outshape.erase(begin(outshape));

  OutputMetaData meta;
  meta.shape = outshape;
  meta.dtype = self.scalar_type();

  return {meta};
}

OutputMetaDataVector Avgpool2dBwdMeta(const at::Stack& stack) {
  auto self = stack.at(1).toTensor();

  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();

  return {meta};
}

SharedMetaDataVector AvgPool2dBwdSharedMeta(const at::Stack& stack) {
  return AvgPoolBwdSharedMeta(stack, "avg_pool_2d_bwd");
}

void Avgpool2dBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = Fillavgpool2dParamsBwd(stack, size);
  auto meta = Avgpool2dBwdMeta(stack)[0];
  if (stack_tensor(stack, 0).dim() == 4) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  } else {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHC,
         synapse_helpers::layouts::SynapseLayoutFormat::WHC},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHC});
  }
  std::vector<synTensor> grad = {syn_in(0)};
  CreateShapeTensorInput(graph, meta.dtype, meta.shape, grad);
  auto avg_pool = BuildOp(
      graph,
      get_guid_with_precision("avg_pool_2d_bwd", meta.dtype),
      std::move(grad),
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(avg_pool[0]);
}

SharedMetaDataVector AvgPool2dFwdSharedMeta(const at::Stack& stack) {
  return Input0SharedMeta(stack, "avg_pool_2d_fwd");
}

void Avgpool2dFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = Fillavgpool2dParamsFwd(stack, size);
  auto meta = Avgpool2dMeta(stack)[0];
  const auto rank = stack_tensor(stack, 0).dim();

  if (rank == 4) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  } else if (rank == 3) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHC,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHC});
  }

  std::vector<synTensor> inputs = {syn_in(0)};
  CreateShapeTensorInput(graph, meta.dtype, meta.shape, inputs);
  auto avg_pool2d = BuildOp(
      graph,
      GetGuid(),
      std::move(inputs),
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(avg_pool2d[0]);
}

} // namespace habana

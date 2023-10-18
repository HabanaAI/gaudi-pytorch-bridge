/*******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/_adaptive_avg_pool2d.h"
#include "generated/backend/_adaptive_avg_pool2d_backward.h"
#include "generated/backend/adaptive_avg_pool2d.h"

namespace habana {

std::shared_ptr<void> FillAdaptiveAvgPool2dParamsFwd(
    const at::Stack& stack,
    size_t& size) {
  const auto output_size = stack[1].toIntList().vec();
  PARAMS_STUB(ns_AdaptiveAvgPool::Params);
  params->outputHeight = output_size[0];
  params->outputWidth = output_size[1];
  return params;
}

std::shared_ptr<void> FillAdaptiveAvgPool2dParamsBwd(
    const at::Stack& stack,
    size_t& size) {
  const auto input = stack_tensor(stack, 1);
  PARAMS_STUB(ns_AdaptiveAvgPool::Params);
  params->outputHeight = input.size(-2);
  params->outputWidth = input.size(-1);
  return params;
}

OutputMetaDataVector AdaptiveAvgPool2dMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const auto output_size = stack[1].toIntList().vec();
  const auto input_size = self.dim();
  TORCH_CHECK(
      input_size == 4 || input_size == 3,
      "AdaptiveAvgPool2d expects input rank to be 4 or 3, but got size ",
      input_size);

  const int64_t output_H = output_size[0];
  const int64_t output_W = output_size.size() == 1 ? output_H : output_size[1];

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = (self.dim() == 4)
      ? std::vector<int64_t>{self.size(0), self.size(1), output_H, output_W}
      : std::vector<int64_t>{self.size(0), output_H, output_W};
  return {meta};
}

OutputMetaDataVector AdaptiveAvgPool2dBwdMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.dtype = stack_tensor(stack, 0).scalar_type();
  meta.shape = stack_tensor(stack, 1).sizes().vec();
  return {meta};
}

void AdaptiveAvgPool2dFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = FillAdaptiveAvgPool2dParamsFwd(stack, size);
  auto meta = AdaptiveAvgPool2dMeta(stack)[0];

  if (stack_tensor(stack, 0).dim() == 4) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  }

  auto adaptive_avg_pool = BuildOp(
      graph,
      GetGuid(),
      {syn_in(0)},
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(adaptive_avg_pool[0]);
}

void AdaptiveAvgPool2dBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = FillAdaptiveAvgPool2dParamsBwd(stack, size);
  auto meta = AdaptiveAvgPool2dBwdMeta(stack)[0];

  if (stack_tensor(stack, 0).dim() == 4) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  }

  std::vector<synTensor> grad = {syn_in(0)};
  this->CreateShapeTensorInput(graph, meta.dtype, meta.shape, grad);
  auto adaptive_avg_pool = BuildOp(
      graph,
      get_guid_with_precision("adaptive_avg_pool_2d_bwd", meta.dtype),
      std::move(grad),
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(adaptive_avg_pool[0]);
}
} // namespace habana

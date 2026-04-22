/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include "generated/backend/_adaptive_avg_pool2d.h"
#include "generated/backend/_adaptive_avg_pool2d_backward.h"
#include "generated/backend/adaptive_avg_pool2d.h"
#include "habana_helpers/conversion.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {

FillParamsT FillAdaptiveAvgPool2dParamsFwd(const at::Stack& stack) {
  const auto output_size = stack[1].toIntList();
  PARAMS_STUB(ns_AdaptiveAvgPool::Params);
  params->outputHeight = safe_convert<int>(output_size[0]);
  params->outputWidth = safe_convert<int>(output_size[1]);
  return paramsT;
}

FillParamsT FillAdaptiveAvgPool2dParamsBwd(const at::Stack& stack) {
  const auto input = stack_tensor(stack, 1);
  PARAMS_STUB(ns_AdaptiveAvgPool::Params);
  params->outputHeight = safe_convert<int>(input.size(-2));
  params->outputWidth = safe_convert<int>(input.size(-1));
  return paramsT;
}

OutputMetaDataVector AdaptiveAvgPool2dMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const auto output_size = stack[1].toIntList().vec();
  const auto input_size = self.dim();
  HABANA_ASSERT(
      input_size == 4 || input_size == 3,
      "AdaptiveAvgPool2d expects input rank to be 4 or 3, but got size ",
      input_size);

  const int64_t output_H = output_size[0];
  const int64_t output_W = output_size.size() == 1 ? output_H : output_size[1];

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.dtype = self.scalar_type();
  meta.shape = (self.dim() == 4)
      ? std::vector<int64_t>{self.size(0), self.size(1), output_H, output_W}
      : std::vector<int64_t>{self.size(0), output_H, output_W};
  return metaVec;
}

OutputMetaDataVector AdaptiveAvgPool2dBwdMeta(const at::Stack& stack) {
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.dtype = stack_tensor(stack, 0).scalar_type();
  meta.shape = stack_tensor(stack, 1).sizes().vec();
  return metaVec;
}

SharedMetaDataVector AdaptiveAvgPool2dFwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return Input0SharedMeta(stack, "adaptive_avg_pool_2d_fwd");
}

void AdaptiveAvgPool2dFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& params = FillAdaptiveAvgPool2dParamsFwd(stack);
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
      params.ptr(),
      params.size());

  syn_out(0) = std::move(adaptive_avg_pool[0]);
}

SharedMetaDataVector AdaptiveAvgPool2dBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return AdaptiveBwdSharedMeta(stack, "complex_adaptive_avg_pool_2d_bwd");
}

void AdaptiveAvgPool2dBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = AdaptiveAvgPool2dBwdMeta(stack)[0];

  if (stack_tensor(stack, 0).dim() == 4) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  } else {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHN});
  }
  auto adaptive_avg_pool = BuildOp(
      graph, GetGuid(), {syn_in(0), syn_in(1)}, {{meta.shape, meta.dtype, 0}});

  syn_out(0) = std::move(adaptive_avg_pool[0]);
}
} // namespace habana

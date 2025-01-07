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
#include "generated/backend/_adaptive_avg_pool3d.h"
#include "generated/backend/_adaptive_avg_pool3d_backward.h"
#include "generated/backend/adaptive_avg_pool3d.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {

std::shared_ptr<void> FillAdaptiveAvgPool3dParamsFwd(
    const at::Stack& stack,
    size_t& size) {
  const auto outputSize = stack[1].toIntList().vec();
  PARAMS_STUB(ns_AdaptiveAvgPool3D::Params);
  params->outputBatch = outputSize[0];
  params->outputHeight =
      outputSize.size() == 1 ? params->outputBatch : outputSize[1];
  params->outputWidth =
      outputSize.size() == 1 ? params->outputBatch : outputSize[2];
  return params;
}

OutputMetaDataVector AdaptiveAvgPool3dMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const auto inputSize = self.dim();
  const auto outputSize = stack[1].toIntList().vec();
  TORCH_CHECK(
      inputSize == 5 || inputSize == 4,
      "AdaptiveAvgPool3d expects input rank to be 4 or 3, but got size ",
      inputSize);

  const int64_t output_N = outputSize[0];
  const int64_t output_H = outputSize.size() == 1 ? output_N : outputSize[1];
  const int64_t output_W = outputSize.size() == 1 ? output_N : outputSize[2];
  std::vector<int64_t> outshape;
  if (inputSize == 5)
    outshape = {self.size(0), self.size(1), output_N, output_H, output_W};
  else
    outshape = {self.size(0), output_N, output_H, output_W};
  OutputMetaData meta;
  meta.shape = outshape;
  meta.dtype = self.scalar_type();
  return {meta};
}

OutputMetaDataVector AdaptiveAvgPool3dBwdMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 1);
  OutputMetaData meta;
  meta.shape = input.sizes().vec();
  meta.dtype = input.scalar_type();
  return {meta};
}

SharedMetaDataVector AdaptiveAvgPool3dFwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return Input0SharedMeta(stack, "adaptive_avg_pool_3d_fwd");
}

void AdaptiveAvgPool3dFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = FillAdaptiveAvgPool3dParamsFwd(stack, size);
  auto meta = AdaptiveAvgPool3dMeta(stack)[0];
  const auto rank = stack_tensor(stack, 0).dim();
  if (rank == 4) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDC},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDC});
  } else if (rank == 5) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN});
  }

  auto adaptiveAvgPool = BuildOp(
      graph,
      GetGuid(),
      {syn_in(0)},
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(adaptiveAvgPool[0]);
}

SharedMetaDataVector AdaptiveAvgPool3dBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return AdaptiveBwdSharedMeta(stack, "complex_adaptive_avg_pool_3d_bwd");
}

void AdaptiveAvgPool3dBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = OutputMeta(stack)[0];
  std::vector<synTensor> inputs = {syn_in(0), syn_in(1)};
  const auto rank = stack_tensor(stack, 0).dim();
  if (rank == 4) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  } else if (rank == 5) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN});
  }
  auto adaptiveAvgPool = BuildOp(
      graph, GetGuid(), std::move(inputs), {{meta.shape, meta.dtype, 0}});
  syn_out(0) = std::move(adaptiveAvgPool[0]);
}

} // namespace habana
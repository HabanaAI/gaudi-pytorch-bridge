/**
 * Copyright (c) 2025 Intel Corporation
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

#include "generated/backend/adaptive_max_pool2d.h"
#include "generated/backend/adaptive_max_pool3d.h"

namespace habana {

at::ScalarType computeKernelIndexType(const at::Tensor& self) {
  return self.scalar_type() == at::kFloat ? torch::kUInt8 : torch::kInt16;
}

OutputMetaDataVector AdaptiveMaxPool2DMeta(const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  const auto& output_size = stack.at(1).toIntVector();

  TORCH_CHECK(
      self.dim() > 2,
      "AdaptiveMaxPool2D requires input tensor with at least 3 dimensions, but got ",
      self.dim());

  std::vector<int64_t> output_shape(self.sizes().vec());
  output_shape.at(output_shape.size() - 2) = output_size[0];
  output_shape.at(output_shape.size() - 1) = output_size[1];

  OutputMetaDataVector meta(2);
  meta[0].shape = output_shape;
  meta[0].dtype = self.scalar_type();
  meta[1].shape = output_shape;
  meta[1].dtype = at::kLong;

  return meta;
}

OutputMetaDataVector AdaptiveMaxPool3DMeta(const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  const auto& output_size = stack.at(1).toIntVector();

  TORCH_CHECK(
      self.dim() > 3,
      "AdaptiveMaxPool3D requires input tensor with at least 4 dimensions, but got ",
      self.dim());

  std::vector<int64_t> output_shape(self.sizes().vec());
  output_shape.at(output_shape.size() - 3) = output_size[0];
  output_shape.at(output_shape.size() - 2) = output_size[1];
  output_shape.at(output_shape.size() - 1) = output_size[2];

  OutputMetaDataVector meta(2);
  meta[0].shape = output_shape;
  meta[0].dtype = self.scalar_type();
  meta[1].shape = output_shape;
  meta[1].dtype = at::kLong;

  return meta;
}

FillParamsT FillAdaptiveMaxPool2DParams(const at::Stack& stack) {
  const auto& output_size = stack.at(1).toIntVector();

  PARAMS_STUB(ns_AdaptiveAvgPool::Params);

  params->outputHeight = output_size[0];
  params->outputWidth = output_size[1];

  return paramsT;
}

FillParamsT FillAdaptiveMaxPool3DParams(const at::Stack& stack) {
  const auto& output_size = stack.at(1).toIntVector();

  PARAMS_STUB(ns_AdaptiveAvgPool3D::Params);

  params->outputBatch = output_size[0];
  params->outputHeight = output_size[1];
  params->outputWidth = output_size[2];

  return paramsT;
}

synapse_helpers::layouts::SynapseLayoutFormat getSynapseLayoutFormat(
    const int rank,
    const bool is_2d) {
  synapse_helpers::layouts::SynapseLayoutFormat result =
      synapse_helpers::layouts::SynapseLayoutFormat::INVALID;

  if (rank == 5) {
    result = synapse_helpers::layouts::SynapseLayoutFormat::WHDCN;
  } else if (rank == 4) {
    result = is_2d ? synapse_helpers::layouts::SynapseLayoutFormat::WHCN
                   : synapse_helpers::layouts::SynapseLayoutFormat::WHDC;
  } else if (rank == 3) {
    result = synapse_helpers::layouts::SynapseLayoutFormat::WHC;
  }

  return result;
};

void AdaptiveMaxPool::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  const auto index_type = computeKernelIndexType(self);
  const auto& meta = OutputMeta(stack);
  const auto& params = FillParams(stack);

  const auto synapse_layout_format = getSynapseLayoutFormat(
      self.dim(), params.size() == sizeof(ns_AdaptiveAvgPool::Params));
  SetSynapseLayouts(
      {synapse_layout_format}, {synapse_layout_format, synapse_layout_format});

  auto maxpool = BuildOp(
      graph,
      GetGuid(),
      {syn_in(0)},
      {{meta[1].shape, index_type}, {meta[0].shape, meta[0].dtype, 0}},
      params.ptr(),
      params.size());

  syn_out(0) = std::move(maxpool[1]);
  syn_out(1) = BuildCast(
      this,
      graph,
      maxpool.at(0).get(),
      meta[1].shape,
      index_type,
      at::kLong,
      1);
}

SharedMetaDataVector AdaptiveMaxPoolCommonSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& self = stack_tensor(stack, 0);
  const auto rank = self.dim();
  const auto dtype = self.scalar_type();
  const auto index_type = computeKernelIndexType(self);

  SharedMetaData maxPoolWithIndicesSharedMeta{guid};
  maxPoolWithIndicesSharedMeta.inputs_data.emplace_back(rank, dtype);
  maxPoolWithIndicesSharedMeta.outputs_data = {
      {rank, index_type}, {rank, dtype}};

  return {maxPoolWithIndicesSharedMeta};
}

SharedMetaDataVector AdaptiveMaxPool2DSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return AdaptiveMaxPoolCommonSharedMeta(stack, "adaptive_max_pool_2d_fwd");
}

SharedMetaDataVector AdaptiveMaxPool3DSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return AdaptiveMaxPoolCommonSharedMeta(stack, "adaptive_max_pool_3d_fwd");
}

} // namespace habana

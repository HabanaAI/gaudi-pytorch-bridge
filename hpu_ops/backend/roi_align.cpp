/**
 * Copyright (c) 2021-2026 Intel Corporation
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

#include "hpu_ops/roi_align.h"
#include "backend/habana_operator.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

static auto PrepareRois(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor syn_rois,
    const std::vector<int64_t>& rois_shape,
    const at::ScalarType& dtype) {
  using namespace std::literals;
  auto rois_outputs = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("prepare_rois_fwd"sv, dtype),
       {syn_rois},
       {{{rois_shape[0], rois_shape[1] - 1}, at::ScalarType::Float},
        {{rois_shape[0]}, at::ScalarType::Int}}});

  return rois_outputs;
}

OutputMetaDataVector ComputeRoiAlignMetadata(const at::Stack& stack) {
  auto dim0 = stack[1].toTensor().size(0);
  auto dim1 = stack[0].toTensor().size(1);
  auto dim2 = stack[3].toInt();
  auto dim3 = stack[4].toInt();

  return {OutputMetaData(
      stack[0].toTensor().scalar_type(), {dim0, dim1, dim2, dim3})};
}

RoiAlign::RoiAlign(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "roialign_fwd", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(ComputeRoiAlignMetadata);
}

void RoiAlign::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "RoiAlign::AddNode");
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto rois = stackGetter.getNextInput<TensorsPair>();
  auto spatial_scale = stackGetter.getNextInput<double>();
  stackGetter.getNextInput<long>();
  stackGetter.getNextInput<long>();
  auto sampling_ratio = stackGetter.getNextInput<long>();
  auto aligned = stackGetter.getNextInput<bool>();

  const auto output_meta = ComputeRoiAlignMetadata(stack)[0];
  const auto& dtype = output_meta.dtype;
  const auto& output_shape = output_meta.shape;

  auto rois_outputs =
      PrepareRois(this, graph, rois.syn_t, rois.pt_t.sizes().vec(), dtype);

  ns_RoiAlignKernel::ParamsAlignment roi_params{};
  roi_params.mode = RoiAlignMode_t::ROI_ALIGN_AVG;
  HABANA_ASSERT(
      sampling_ratio <= std::numeric_limits<int>::max(),
      "Sampling ratio exceeds maximum value for int.");
  roi_params.sampling_ratio = static_cast<int>(sampling_ratio);
  roi_params.spatial_scale = static_cast<float>(spatial_scale);
  roi_params.aligned = aligned;

  SetSynapseLayouts(
      {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::XR,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
      {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});

  std::vector<synTensor> inputs = {
      input.syn_t, rois_outputs[0].get(), rois_outputs[1].get()};
  CreateShapeTensorInput(graph, dtype, output_shape, inputs);

  auto output = OpBackend::BuildOp(
      graph,
      guid_,
      std::move(inputs),
      {{output_shape, dtype, 0}},
      &roi_params,
      sizeof(roi_params));

  syn_out(0) = std::move(output[0]);
}

OutputMetaDataVector ComputeRoiAlignBackwardMetadata(const at::Stack& stack) {
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape.reserve(4);
  for (size_t i = 5; i < 9; ++i) {
    meta.shape.push_back(stack[i].toInt());
  }
  meta.dtype = stack[0].toTensor().scalar_type();
  return metaVec;
}

RoiAlignBackward::RoiAlignBackward(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "roialign_bwd", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(ComputeRoiAlignBackwardMetadata);
}

void RoiAlignBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "RoiAlignBackward::AddNode");
  auto grad = stackGetter.getNextInput<TensorsPair>();
  auto rois = stackGetter.getNextInput<TensorsPair>();
  auto spatial_scale = stackGetter.getNextInput<double>();
  stackGetter.getNextInput<long>();
  stackGetter.getNextInput<long>();
  stackGetter.getNextInput<long>();
  stackGetter.getNextInput<long>();
  stackGetter.getNextInput<long>();
  stackGetter.getNextInput<long>();
  auto sampling_ratio = stackGetter.getNextInput<long>();
  auto aligned = stackGetter.getNextInput<bool>();

  const auto rois_shape = rois.pt_t.sizes().vec();

  const auto output_meta = ComputeRoiAlignBackwardMetadata(stack)[0];
  const auto& dtype = output_meta.dtype;
  const auto& output_shape = output_meta.shape;

  auto rois_outputs = PrepareRois(this, graph, rois.syn_t, rois_shape, dtype);

  std::vector<synTensor> inputs = {
      grad.syn_t, rois_outputs[0].get(), rois_outputs[1].get()};
  std::vector<synapse_helpers::tensor> quad_tree;

  CreateShapeTensorInput(graph, dtype, output_shape, inputs);

  ns_RoiAlignBwdKernel::ParamsIsValidCount roi_params{};
  roi_params.mode = RoiAlignMode_t::ROI_ALIGN_AVG;
  HABANA_ASSERT(
      sampling_ratio <= std::numeric_limits<int>::max(),
      "Sampling ratio exceeds maximum value for int.");
  roi_params.sampling_ratio = static_cast<int>(sampling_ratio);
  roi_params.spatial_scale = static_cast<float>(spatial_scale);
  roi_params.aligned = static_cast<int>(aligned);
  roi_params.isValidCount = false;

  SetSynapseLayouts(
      {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::VN,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::NSB,
       synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
      {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});

  auto output = BuildOp(
      graph,
      guid_,
      std::move(inputs),
      {{output_shape, dtype, 0}},
      &roi_params,
      sizeof(roi_params));

  syn_out(0) = std::move(output[0]);
}

} // namespace habana

static const auto& RoiAlignKernelRegistry =
    habana::KernelRegistry()
        .REGISTER_HPU_BACKEND("torchvision::roi_align", habana::RoiAlign)
        .REGISTER_HPU_BACKEND(
            "torchvision::_roi_align_backward",
            habana::RoiAlignBackward);

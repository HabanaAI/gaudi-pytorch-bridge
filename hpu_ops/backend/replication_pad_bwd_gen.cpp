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

#include "hpu_ops/common/replication_pad.h"

namespace habana {
OutputMetaDataVector ReplicationPadBwdMeta(const at::Stack& stack) {
  auto self = stack.at(1).toTensor();
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  return metaVec;
}

FillParamsT FillReplicationPad1dBwdParams(const at::Stack& stack) {
  return FillPadFwdBwdParams(stack, pad1D, true);
}

FillParamsT FillReplicationPad2dBwdParams(const at::Stack& stack) {
  return FillPadFwdBwdParams(stack, pad2D, true);
}

FillParamsT FillReplicationPad3dBwdParams(const at::Stack& stack) {
  return FillPadFwdBwdParams(stack, pad3D, true);
}

std::vector<synapse_helpers::tensor> CommonReplicationPadBwd(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    const FillParamsT& params,
    synTensor input,
    PadType pad) {
  auto meta = op->OutputMeta(stack)[0];
  if (habana::ShapeInference::GetCurrentPass() ==
      habana::ShapeInfo::InferencePass::MAX_SHAPE) {
    const synapse_helpers::tensor& synTensorGrad = op->ReadSynInput(0);
    std::vector<int64_t> currentMaxShape =
        std::get<1>(habana::ShapeInference::GetMinMaxShape(synTensorGrad.id()));
    auto outputShapeExpectedMax =
        ComputePadOutputShape({stack.begin() + 1, stack.end()}, pad)[0];

    auto currentMaxShapeSize = currentMaxShape.size();
    for (size_t dim = 0; dim < currentMaxShapeSize; dim++) {
      HABANA_ASSERT(
          (currentMaxShape[dim] <= outputShapeExpectedMax[dim]),
          "Dim (%d) size (%d) in max pass is greater than expected size (%d)",
          dim,
          currentMaxShape[dim],
          outputShapeExpectedMax[dim]);
    }
  }

  using namespace std::literals;
  return OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("pad_bwd"sv, meta.dtype),
       {input},
       {{meta.shape, meta.dtype, 0}},
       params.ptr(),
       params.size()});
}

void ReplicationPad1dBwdOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto params = FillParams(stack);
  auto padOutput =
      CommonReplicationPadBwd(this, graph, stack, params, syn_in(0), pad1D);
  syn_out(0) = std::move(padOutput[0]);
}

void ReplicationPad2dBwdOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto params = FillParams(stack);
  auto padOutput =
      CommonReplicationPadBwd(this, graph, stack, params, syn_in(0), pad2D);
  syn_out(0) = std::move(padOutput[0]);
}

void ReplicationPad3dBwdOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto params = FillParams(stack);
  auto padOutput =
      CommonReplicationPadBwd(this, graph, stack, params, syn_in(0), pad3D);
  syn_out(0) = std::move(padOutput[0]);
}
} // namespace habana

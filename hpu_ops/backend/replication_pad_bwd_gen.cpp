/******************************************************************************
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

#include "hpu_ops/common/replication_pad.h"

namespace habana {
OutputMetaDataVector ReplicationPadBwdMeta(const at::Stack& stack) {
  auto self = stack.at(1).toTensor();
  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  return {meta};
}

std::shared_ptr<void> FillReplicationPad1dBwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillPadFwdBwdParams(stack, pad1D, size, true);
}

std::shared_ptr<void> FillReplicationPad2dBwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillPadFwdBwdParams(stack, pad2D, size, true);
}

std::shared_ptr<void> FillReplicationPad3dBwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillPadFwdBwdParams(stack, pad3D, size, true);
}

std::vector<synapse_helpers::tensor> CommonReplicationPadBwd(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    std::shared_ptr<void> params,
    size_t& size,
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
    for (size_t dim = 0; dim < currentMaxShapeSize; dim++)
      TORCH_CHECK(
          (currentMaxShape[dim] <= outputShapeExpectedMax[dim]),
          "Dim (%d) size (%d) in max pass is greater than expected size (%d)",
          dim,
          currentMaxShape[dim],
          outputShapeExpectedMax[dim]);
  }

  return op->BuildNode(
      op,
      graph,
      {get_guid_with_precision("pad_bwd", meta.dtype),
       {input},
       {{meta.shape, meta.dtype, 0}},
       params.get(),
       size});
}

void ReplicationPad1dBwdOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  auto params = FillParams(stack, size);
  auto padOutput = CommonReplicationPadBwd(
      this, graph, stack, params, size, syn_in(0), pad1D);
  syn_out(0) = std::move(padOutput[0]);
}

void ReplicationPad2dBwdOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  auto params = FillParams(stack, size);
  auto padOutput = CommonReplicationPadBwd(
      this, graph, stack, params, size, syn_in(0), pad2D);
  syn_out(0) = std::move(padOutput[0]);
}

void ReplicationPad3dBwdOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  auto params = FillParams(stack, size);
  auto padOutput = CommonReplicationPadBwd(
      this, graph, stack, params, size, syn_in(0), pad3D);
  syn_out(0) = std::move(padOutput[0]);
}
} // namespace habana

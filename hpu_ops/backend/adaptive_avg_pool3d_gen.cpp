/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/_adaptive_avg_pool3d.h"
#include "generated/backend/adaptive_avg_pool3d.h"
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

void AdaptiveAvgPool3dFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = FillAdaptiveAvgPool3dParamsFwd(stack, size);
  auto meta = AdaptiveAvgPool3dMeta(stack)[0];
  auto intermediateOutShape = meta.shape;
  auto self = stack_tensor(stack, 0);
  auto reshapeRequired = (self.dim() == 4);
  std::vector<synTensor> inputs = {syn_in(0)};
  std::vector<synapse_helpers::tensor> expandResult;
  c10::optional<int> finalIndex =
      reshapeRequired ? c10::nullopt : c10::make_optional<int>(0);

  if (reshapeRequired) {
    std::vector<int64_t> inputExpandedShape = {1};
    const auto& selfShape = self.sizes().vec();
    inputExpandedShape.insert(
        std::end(inputExpandedShape),
        std::begin(selfShape),
        std::end(selfShape));
    intermediateOutShape.insert(std::begin(intermediateOutShape), 1);
    synAxisParams expandParams{4};
    auto expandedInput = BuildOp(
        graph,
        "expand_dims",
        std::move(inputs),
        {{inputExpandedShape, meta.dtype}},
        &expandParams,
        sizeof(expandParams));
    expandResult.push_back(std::move(expandedInput[0]));
    inputs = {expandResult[0].get()};
  }

  auto adaptiveAvgPool = BuildOp(
      graph,
      GetGuid(),
      std::move(inputs),
      {{intermediateOutShape, meta.dtype, finalIndex}},
      params.get(),
      size);

  if (reshapeRequired) {
    auto reshapedAdaptiveAvgPool = ReshapeHelper(
        graph, adaptiveAvgPool[0].get(), meta.shape, meta.dtype, 0);
    syn_out(0) = std::move(reshapedAdaptiveAvgPool);
  } else {
    syn_out(0) = std::move(adaptiveAvgPool[0]);
  }
}

} // namespace habana
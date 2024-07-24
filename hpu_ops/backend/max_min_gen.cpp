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

#include "backend/helpers/create_tensor.h"
#include "generated/backend/max.h"
#include "generated/backend/min.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {

OutputMetaDataVector ReduceMinMaxMeta(const at::Stack& stack) {
  const auto self = stack.at(0).toTensor();

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = {};
  return {meta};
}

std::shared_ptr<void> FillMinMaxParams(const at::Stack&, size_t& size) {
  PARAMS_STUB(ns_Reduction::ParamsV2);
  params->reductionDimensionMask = 0;
  params->keepDim = false;
  return params;
}

sizes_vec MinMaxOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  int64_t dim = stack.at(1).toInt();
  bool keepdim = stack.at(2).toBool();
  auto shape = ReductionOutputShape(self, dim, keepdim)[0];
  return {shape, shape};
}

OutputMetaDataVector MinMaxMeta(const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  auto outputShape = MinMaxOutputShape(stack)[0];
  auto memoryFormat = self.suggest_memory_format();

  OutputMetaData metaMinMax;
  metaMinMax.shape = outputShape;
  metaMinMax.mem_format = memoryFormat;
  metaMinMax.dtype = self.scalar_type();

  OutputMetaData metaIndices;
  metaIndices.shape = outputShape;
  metaIndices.mem_format = memoryFormat;
  metaIndices.dtype = c10::ScalarType::Long;

  return {metaMinMax, metaIndices};
}

std::shared_ptr<void> FillMinMaxDimParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_Reduction::Params);
  auto dim = stack.at(1).toInt();
  dim = (dim >= 0) ? static_cast<int>(stack.at(0).toTensor().dim()) - 1 - dim
                   : -(dim + 1);

  params->reductionDimension = dim;
  return params;
}

void MinMaxOut::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dim = stack.at(1).toInt();
  auto keepdim = stack.at(2).toBool();

  auto meta = MinMaxMeta(stack);
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {meta[0].shape, meta[0].dtype, 0}, {meta[1].shape, meta[1].dtype, 1}};

  auto params = FillReductionParams(self.dim(), {dim}, keepdim);

  auto result = OpBackend::BuildNode(
      this,
      graph,
      {guid_, {syn_in(0)}, std::move(output_attrs), &params, sizeof(params)});

  syn_out(0) = std::move(result[0]);
  syn_out(1) = std::move(result[1]);
}

} // namespace habana

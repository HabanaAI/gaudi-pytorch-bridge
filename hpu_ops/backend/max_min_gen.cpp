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
  auto shape = MinMaxOutputShape(stack)[0];
  auto meta = MinMaxMeta(stack);
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {meta[0].shape, meta[0].dtype, 0}, {meta[1].shape, meta[1].dtype, 1}};

  auto reduce_max = HandleReductionDimAndKeepdim(
      this, graph, self, {syn_in(0)}, dim, keepdim, guid_, output_attrs);

  syn_out(0) = std::move(reduce_max[0]);
  syn_out(1) = std::move(reduce_max[1]);
}

void MaxDimOp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto dim = stack.at(1).toInt();
  bool keepdim = stack.at(2).toBool();
  auto meta = MinMaxMeta(stack);
  size_t size = 0;
  const auto& params = FillMinMaxDimParams(stack, size);

  if (self.dim() == 0) {
    auto res = BuildOp(
        graph, "identity", {syn_in(0)}, {{meta[0].shape, meta[0].dtype, 0}});
    auto index =
        ConstantHelper(graph, /*val=*/0, meta[0].dtype, meta[1].shape, 1);
    syn_out(0) = std::move(res[0]);
    syn_out(1) = std::move(index);
  } else {
    if (keepdim) {
      auto max_dim = BuildOp(
          graph,
          guid_,
          {syn_in(0)},
          {{meta[0].shape, meta[0].dtype, 0},
           {meta[1].shape, meta[1].dtype, 1}},
          params.get(),
          size);

      syn_out(0) = std::move(max_dim[0]);
      syn_out(1) = std::move(max_dim[1]);
    } else {
      auto shape = self.sizes().vec();
      dim = c10::maybe_wrap_dim(dim, self.dim(), true);
      shape[dim] = 1;

      auto max_dim = BuildOp(
          graph,
          guid_,
          {syn_in(0)},
          {{shape, meta[0].dtype}, {shape, meta[1].dtype}},
          params.get(),
          size);
      auto max = ReshapeHelper(
          graph, max_dim[0].get(), meta[0].shape, meta[0].dtype, 0);
      auto max_indices = ReshapeHelper(
          graph, max_dim[1].get(), meta[1].shape, meta[1].dtype, 1);

      syn_out(0) = std::move(max);
      syn_out(1) = std::move(max_indices);
    }
  }
}
} // namespace habana

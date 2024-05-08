/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/median.h"
#include "hpu_ops/median_slice_util.h"
#include "hpu_ops/topk_util.h"

namespace habana {
constexpr size_t index_of_self = 0;
constexpr size_t index_of_reduction_axis = 1;
constexpr size_t index_of_keepdim = 2;
constexpr int descending_order = 0;

OutputMetaDataVector MedianOutputMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = {};
  meta.dtype = stack_tensor(stack, 0).scalar_type();
  return {meta};
}

sizes_vec MediandimOutputShape(const at::Stack& stack) {
  auto self = stack.at(index_of_self).toTensor();
  auto self_size = self.sizes().vec();
  int64_t reduction_axis = c10::maybe_wrap_dim(
      stack[index_of_reduction_axis].toInt(),
      self.dim(),
      /*wrap_scalar=*/true);

  bool keepdim = stack[index_of_keepdim].toBool();
  std::vector<int64_t> outshape = {self_size};
  if (outshape.size() == 0) {
    return {outshape, outshape};
  }

  if (keepdim)
    outshape[reduction_axis] = 1;
  else {
    std::vector<int64_t>::iterator itr = outshape.begin() + reduction_axis;
    outshape.erase(itr);
  }
  return {outshape, outshape};
}

OutputMetaDataVector MedianDimOutputMeta(const at::Stack& stack) {
  auto medianDimShapes = MediandimOutputShape(stack);
  auto self = stack_tensor(stack, index_of_self);

  OutputMetaData valuesMeta, indicesMeta;

  valuesMeta.shape = medianDimShapes[0];
  valuesMeta.dtype = self.scalar_type();

  indicesMeta.shape = medianDimShapes[1];
  indicesMeta.dtype =
      common::IsInt64Supported() ? c10::ScalarType::Long : c10::ScalarType::Int;

  return {valuesMeta, indicesMeta};
}

void Mediandim::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, index_of_self);
  auto self_size = self.sizes().vec();

  if (self_size.size() == 0) {
    auto out_shape = MediandimOutputShape(stack)[0];
    auto out = OpBackend::BuildOp(
        graph, "identity", {syn_in(0)}, {{out_shape, ScalarType(), 0}});
    syn_out(0) = std::move(out[0]);
    auto indices_dtype = common::IsInt64Supported() ? c10::ScalarType::Long
                                                    : c10::ScalarType::Int;
    auto index = ConstantHelper(graph, /*val=*/0, indices_dtype, out_shape, 1);
    syn_out(1) = std::move(index);
    return;
  }

  bool keepdim = stack[index_of_keepdim].toBool();
  int64_t reduction_axis = c10::maybe_wrap_dim(
      stack[index_of_reduction_axis].toInt(),
      self.dim(),
      /*wrap_scalar=*/true);
  auto meta = MedianDimOutputMeta(stack);

  std::vector<int64_t> topk_outshape;
  topk_outshape = self_size;

  auto topk = TopK_Helper(
      this,
      graph,
      {syn_in(index_of_self)},
      reduction_axis,
      topk_outshape,
      descending_order,
      self.ndimension(),
      topk_outshape[reduction_axis],
      1, /* median variant */
      c10::nullopt);

  std::vector<int64_t> slice_outshape;
  slice_outshape = self_size;
  /* The output tensor will have the single median value along the reduction
     axis. Hence the size along the reduction axis = 1 */
  slice_outshape[reduction_axis] = 1;

  auto median_value = Median_Slice_Helper(
      this,
      graph,
      {topk[0].get()},
      slice_outshape,
      meta[0].dtype,
      self_size[reduction_axis],
      self.ndimension(),
      reduction_axis,
      1 /* median variant */,
      keepdim,
      0 /* node index*/);

  auto median_index = Median_Slice_Helper(
      this,
      graph,
      {topk[1].get()},
      slice_outshape,
      meta[1].dtype,
      self_size[reduction_axis],
      self.ndimension(),
      reduction_axis,
      1 /* median variant */,
      keepdim,
      1 /* node index */);

  if (keepdim) {
    syn_out(0) = std::move(median_value[0]);
    syn_out(1) = std::move(median_index[0]);
  } else {
    auto output_shape = MediandimOutputShape(stack)[0];
    auto reshaped_median_value = ReshapeHelper(
        graph, median_value[0].get(), meta[0].shape, meta[0].dtype, 0);

    auto reshaped_median_index = ReshapeHelper(
        graph, median_index[0].get(), meta[1].shape, meta[1].dtype, 1);

    syn_out(0) = std::move(reshaped_median_value);
    syn_out(1) = std::move(reshaped_median_index);
  }
}

} // namespace habana

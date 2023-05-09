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

sizes_vec MedianOutputShape(const at::Stack& stack) {
  static_cast<void>(stack);
  return {{}};
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

void Median::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, index_of_self);
  auto self_size = self.sizes().vec();
  /* N-dimension tensor will be reshaped to 1-dimension tensor.
     Hence the reduction axis is always equal to 0 */
  int64_t reduction_axis = 0;

  std::vector<int64_t> reshape_size = {self.numel()};
  std::vector<int64_t> reshape_outshape = {reshape_size};

  auto reshaped_inp = ReshapeHelper(
      graph, syn_in(index_of_self), reshape_outshape, ScalarType());

  std::vector<int64_t> topk_outshape;
  topk_outshape = {self.numel()};
  auto topk = TopK_Helper(
      this,
      graph,
      {reshaped_inp.get()},
      reduction_axis,
      topk_outshape,
      descending_order,
      0,
      topk_outshape[reduction_axis],
      0 /*median vairiant*/);

  std::vector<int64_t> slice_outshape;
  /* The output is a tensor having single value (i.e. median)*/
  slice_outshape.push_back(1);

  auto median_value = Median_Slice_Helper(
      this,
      graph,
      {topk[0].get()},
      slice_outshape[0],
      ScalarType(),
      self.numel(),
      self.ndimension(),
      reduction_axis,
      0 /*median variant*/,
      false);

  auto output_shape = MedianOutputShape(stack)[0];
  auto median_output = ReshapeHelper(
      graph, median_value[0].get(), output_shape, ScalarType(), 0);
  syn_out(0) = std::move(median_output);
}

void Mediandim::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, index_of_self);
  auto self_size = self.sizes().vec();
  bool keepdim = stack[index_of_keepdim].toBool();
  int64_t reduction_axis = c10::maybe_wrap_dim(
      stack[index_of_reduction_axis].toInt(),
      self.dim(),
      /*wrap_scalar=*/true);

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
      1 /* median variant */);

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
      ScalarType(),
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
      c10::ScalarType::Int,
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
        graph, median_value[0].get(), output_shape, ScalarType(), 0);

    auto reshaped_median_index = ReshapeHelper(
        graph, median_index[0].get(), output_shape, c10::ScalarType::Int, 1);

    syn_out(0) = std::move(reshaped_median_value);
    syn_out(1) = std::move(reshaped_median_index);
  }
}

} // namespace habana

/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/hpu_op.h"
#include "hpu_op_helper.h"
#include "reduction_template.h"

namespace habana {

sizes_vec AllAnyOutputShape(const at::Stack&, bool) {
  return {{}};
}

sizes_vec AllDimOutputShape(const at::Stack& stack, bool) {
  auto shape = stack.at(0).toTensor().sizes().vec();
  const int64_t axis = stack.at(1).toInt();
  auto dim = (axis >= 0) ? axis : stack.at(0).toTensor().dim() + axis;

  shape[dim] = 1;
  if (!stack.at(2).toBool())
    shape.erase(shape.begin() + dim);

  return {shape};
}

void AllDim::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto outshape = self.sizes().vec();

  auto cast_f32 = CastHelper(
      graph, syn_in(0), outshape, self.scalar_type(), c10::ScalarType::Float);

  const int64_t axis = stack.at(1).toInt();
  const bool keepdim = stack.at(2).toBool();
  auto dim = (axis >= 0) ? axis : stack.at(0).toTensor().dim() + axis;

  auto out_shape = AllDimOutputShape(stack, true)[0];

  auto reduce_prod = HandleReductionDimAndKeepdim(
      this,
      graph,
      self,
      {cast_f32.get()},
      dim,
      keepdim,
      "reduce_prod_fwd_f32",
      {{out_shape, ScalarType()}});

  auto cast_i8 = CastHelper(
      graph,
      reduce_prod[0].get(),
      out_shape,
      c10::ScalarType::Float,
      c10::ScalarType::Bool,
      0);

  syn_out(0) = std::move(cast_i8);
}

void All::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  const auto& outshape = stack_tensor(stack, 0).sizes();

  auto cast_f32 = CastHelper(
      graph, syn_in(0), outshape, self.scalar_type(), c10::ScalarType::Float);

  auto self_size = self.sizes();
  auto reshape_size = std::accumulate(
      std::begin(self_size),
      std::end(self_size),
      1,
      std::multiplies<int64_t>());

  std::vector<int64_t> reshape_outshape = {reshape_size};
  auto reshape = ReshapeHelper(
      graph, cast_f32.get(), reshape_outshape, c10::ScalarType::Float);

  size_t size = 0;
  PARAMS_STUB(ns_Reduction::Params);
  params->reductionDimension = 0;

  auto reduce_prod = BuildOp(
      graph,
      "reduce_prod_fwd_f32",
      {reshape.get()},
      {{1, c10::ScalarType::Float}},
      params.get(),
      size);
  auto out_shape = AllOutputShape(stack, true)[0];

  auto cast_i8 = CastHelper(
      graph,
      reduce_prod[0].get(),
      out_shape,
      c10::ScalarType::Float,
      c10::ScalarType::Bool,
      0);

  syn_out(0) = std::move(cast_i8);
}
} // namespace habana

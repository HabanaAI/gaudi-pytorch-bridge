/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/all.h"
#include "hpu_ops/backend/reduction_template.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {
static auto AllCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    const synTensor& syn_in,
    at::IntArrayRef dim,
    bool keepdim,
    at::IntArrayRef final_shape) {
  auto cast_f32 = OpBackend::BuildCast(
      op,
      graph,
      syn_in,
      self.sizes(),
      self.scalar_type(),
      c10::ScalarType::Float);

  op->SetScalarType(at::kFloat);
  auto reduce_prod = HandleReductionDimAndKeepdim(
      op,
      graph,
      self,
      {cast_f32.get()},
      dim,
      keepdim,
      "reduce_prod_fwd_f32",
      {{final_shape, at::kFloat}});

  return OpBackend::BuildCast(
      op, graph, reduce_prod[0].get(), final_shape, at::kFloat, at::kBool, 0);
}

void AllDim::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  const int64_t dim = stack.at(1).toInt();
  const bool keepdim = stack.at(2).toBool();

  auto out = AllCommon(
      this,
      graph,
      self,
      syn_in(0),
      dim,
      keepdim,
      AllAnyDimMeta(stack)[0].shape);
  syn_out(0) = std::move(out);
}

void All::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out = AllCommon(
      this, graph, self, syn_in(0), {}, false, ComputeOutputShapes(stack)[0]);
  syn_out(0) = std::move(out);
}
} // namespace habana

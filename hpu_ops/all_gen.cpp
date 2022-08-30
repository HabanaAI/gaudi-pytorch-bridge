/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/all.h"
#include "hpu_op_helper.h"
#include "reduction_template.h"

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

  auto zero_tensor = OpBackend::BuildConstant(op, graph, 0.f);

  // Use not_equal directly after getting
  // https://jira.habana-labs.com/browse/SW-107386 fixed
  auto equal = OpBackend::BuildNode(
      op,
      graph,
      {"equal_fwd_f32",
       {reduce_prod[0].get(), zero_tensor.get()},
       {{final_shape, c10::ScalarType::Bool}}});

  return OpBackend::BuildNode(
      op,
      graph,
      {"not_fwd_i8",
       {equal[0].get()},
       {{final_shape, c10::ScalarType::Bool, 0}}});
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
      ComputeOutputShapes(stack)[0]);
  syn_out(0) = std::move(out[0]);
}

void All::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out = AllCommon(
      this, graph, self, syn_in(0), {}, false, ComputeOutputShapes(stack)[0]);
  syn_out(0) = std::move(out[0]);
}
} // namespace habana

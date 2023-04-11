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

#include "generated/backend/all.h"
#include "generated/backend/any.h"
#include "habana_kernels/reduction_kernels.h"
#include "hpu_ops/backend/reduction_template.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {
sizes_vec AllAnyOutputShape(const at::Stack&) {
  return {{}};
}

OutputMetaDataVector AllAnyDimMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim = stack.at(1).toInt();
  const bool keepdim = stack.at(2).toBool();

  OutputMetaData meta;
  meta.shape = ReductionOutputShape(self, dim, keepdim)[0];
  meta.dtype = at::kBool;
  return {meta};
}

static synapse_helpers::tensor AnyCommonFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor input,
    const at::Tensor& self,
    const at::IntArrayRef dim,
    const bool keepdim,
    const at::IntArrayRef outshape) {
  // TODO: for integral types, use reduce_sum_fwd_i32 instead
  const auto& dtype = at::kFloat;
  std::unique_ptr<synapse_helpers::tensor> cast;
  if (dtype != self.scalar_type()) {
    cast = std::make_unique<synapse_helpers::tensor>(OpBackend::BuildCast(
        op, graph, input, self.sizes(), self.scalar_type(), dtype));
    if (!op->isMetaMode()) {
      input = cast->get();
    }
  }

  op->SetScalarType(dtype);

  auto abs = OpBackend::BuildNode(
      op, graph, {"abs_fwd_f32", {input}, {{self.sizes().vec()}}});

  auto reduce_sum = HandleReductionDimAndKeepdim(
      op,
      graph,
      self,
      {abs[0].get()},
      dim,
      keepdim,
      "reduce_sum_fwd_f32",
      {{outshape}});

  return OpBackend::BuildCast(
      op, graph, reduce_sum[0].get(), outshape, dtype, at::kBool, 0);
}

void AnyDim::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto dim = stack.at(1).toInt();
  bool keepdim = stack.at(2).toBool();

  auto any_out = AnyCommonFunc(
      this,
      graph,
      syn_in(0),
      self,
      dim,
      keepdim,
      AllAnyDimMeta(stack)[0].shape);
  syn_out(0) = std::move(any_out);
}

void Any::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);

  auto any_out = AnyCommonFunc(
      this, graph, syn_in(0), self, {}, false, ComputeOutputShapes(stack)[0]);
  syn_out(0) = std::move(any_out);
}
} // namespace habana

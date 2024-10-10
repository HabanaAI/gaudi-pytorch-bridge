/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/mean.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {

namespace sh = synapse_helpers;

OutputMetaDataVector ReductionOpMeta(const at::Stack& stack) {
  return ReductionMeta<-1, -1, 1>(stack);
}

OutputMetaDataVector ReductionOpListMeta(const at::Stack& stack) {
  return ReductionMeta<1, 2, 3>(stack);
}

static sh::tensor ReductionOpCommon(
    OpBackend* op,
    sh::graph& graph,
    synTensor input,
    const at::Stack& stack,
    const at::optional<uint8_t>& dim_index,
    const at::optional<uint8_t>& keepdim_index,
    const at::optional<uint8_t>& dtype_index) {
  auto self = stack_tensor(stack, 0);
  auto dtype = get_dtype(stack, dtype_index);
  auto cast = HandleReductionDtype(op, graph, self, input, dtype);
  if (cast.has_value()) {
    input = cast.value().get();
  }

  auto dims = get_dims(stack, dim_index);
  bool keepdim = get_keepdim(stack, keepdim_index);

  int ndims = self.dim();
  auto params = FillReductionParams(ndims, dims, keepdim);
  auto shape = ReductionOutputShape(self, dims, keepdim)[0];
  auto result = OpBackend::BuildNode(
      op,
      graph,
      {op->GetGuid(),
       {std::move(input)},
       {{shape, op->ScalarType(), 0}},
       &params,
       sizeof(params)});
  return std::move(result[0]);
}

void ReductionOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  syn_out(0) = ReductionOpCommon(
      this, graph, syn_in(0), stack, c10::nullopt, c10::nullopt, 1);
}

void ReductionOpList::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  syn_out(0) = ReductionOpCommon(this, graph, syn_in(0), stack, 1, 2, 3);
}
} // namespace habana
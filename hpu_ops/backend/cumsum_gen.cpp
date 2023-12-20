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

#include "generated/backend/cumprod.h"
#include "generated/backend/cumsum.h"

namespace habana {

OutputMetaDataVector CumsumMeta(const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.mem_format = self.suggest_memory_format();

  if (stack.at(2).isNone()) {
    if (isIntegralType(self.scalar_type(), true))
      meta.dtype = c10::ScalarType::Long;
    else
      meta.dtype = self.scalar_type();
  } else
    meta.dtype = stack.at(2).toScalarType();

  return {meta};
}

std::shared_ptr<void> FillCumsumParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_CumSumKernel::Params);
  auto self = stack.at(0).toTensor();
  auto dim = at::maybe_wrap_dim(stack.at(1).toInt(), self.dim(), true);
  params->axis = static_cast<int>(self.sizes().vec().size() - dim - 1);

  return params;
}

void CumsumHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = CumsumMeta(stack)[0];
  at::ScalarType dtype =
      stack.at(2).isNone() ? ScalarType() : stack.at(2).toScalarType();

  if (habana_helpers::is_downcast_to_int_needed(dtype)) {
    dtype = at::ScalarType::Int;
  }

  if (dtype == at::ScalarType::Double) {
    dtype = at::ScalarType::Float;
  }

  if (dtype == at::ScalarType::Bool || dtype == at::ScalarType::Char ||
      dtype == at::ScalarType::Byte) {
    dtype = at::ScalarType::Int;
  }

  if (dtype == ScalarType() || ScalarType() == at::ScalarType::Double) {
    return OpBackend::AddNode(graph, stack);
  }

  auto cast =
      BuildCast(this, graph, syn_in(0), meta.shape, ScalarType(), dtype);
  size_t size = 0;
  const auto& params = FillCumsumParams(stack, size);
  update_guid_dtype(guid_, dtype);

  auto op = BuildOp(
      graph, guid_, {cast.get()}, {{meta.shape, dtype, 0}}, params.get(), size);
  syn_out(0) = std::move(op.at(0));
}

} // namespace habana

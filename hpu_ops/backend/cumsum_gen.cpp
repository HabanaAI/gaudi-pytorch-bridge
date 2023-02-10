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

#include <utility>

#include "generated/backend/cumprod.h"
#include "generated/backend/cumsum.h"

namespace habana {

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
  at::ScalarType dtype =
      stack.at(2).isNone() ? ScalarType() : stack.at(2).toScalarType();

  if (habana_helpers::is_downcast_to_int_needed(dtype)) {
    dtype = at::ScalarType::Int;
  }

  if (dtype == at::ScalarType::Double) {
    dtype = at::ScalarType::Float;
  }

  if (dtype == at::ScalarType::Bool || dtype == at::ScalarType::Char) {
    dtype = at::ScalarType::Int;
  }

  if (dtype == ScalarType() || ScalarType() == at::ScalarType::Double) {
    return OpBackend::AddNode(graph, stack);
  }

  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto cast = CastHelper(graph, syn_in(0), outshape, ScalarType(), dtype);

  size_t size = 0;
  const auto& params = FillCumsumParams(stack, size);
  const std::string& guid = guid_.substr(0, guid_.find_last_of('_') + 1);
  const std::string& cast_to = habana_helpers::name_suffix_from_type(dtype);

  auto op = BuildOp(
      graph,
      guid + cast_to,
      {cast.get()},
      {{outshape, dtype, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(op.at(0));
}

} // namespace habana

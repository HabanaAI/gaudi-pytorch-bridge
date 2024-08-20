/******************************************************************************
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

#include "generated/backend/cumprod.h"
#include "generated/backend/cumsum.h"
#include "hpu_ops/shared_meta_common.h"

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

SharedMetaDataVector FillCumSumSharedMeta(const at::Stack& stack) {
  return FillCumSumProdSharedMeta(stack, "cumsum_fwd");
}

SharedMetaDataVector FillCumProdSharedMeta(const at::Stack& stack) {
  return FillCumSumProdSharedMeta(stack, "cumprod_fwd");
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

  std::optional<synapse_helpers::tensor> cast{};
  bool is_cast_needed = true;
  // If the source and destination datatype are same,
  // then cast is not needed.
  // Different datatypes can map to same precision so
  // this check is necessary. For example, if INT64 is not
  // supported, both Long and INT map to i32 precision in TPC.
  auto from_dtype = habana_helpers::GetPrecisionString(ScalarType());
  auto to_dtype = habana_helpers::GetPrecisionString(dtype);
  if (from_dtype == to_dtype) {
     is_cast_needed = false;
  }

  // Add cast op if necessary
  if (is_cast_needed) {
      cast = BuildCast(this, graph, syn_in(0), meta.shape, ScalarType(), dtype);
  }

  size_t size = 0;
  const auto& params = FillCumsumParams(stack, size);
  update_guid_dtype(guid_, dtype);

  // Get input to cumsum op
  // Can be cast op if cast is enabled.
  auto input_data = (cast.has_value()) ? cast->get() : syn_in(0);

  auto op = BuildOp(
      graph, guid_, {input_data}, {{meta.shape, dtype, 0}}, params.get(), size);
  syn_out(0) = std::move(op.at(0));
}

} // namespace habana

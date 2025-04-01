/*******************************************************************************
 * Copyright (C) 2025 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/index_add.h"

using namespace std::literals;

namespace habana {

std::shared_ptr<void> FillIndexAddParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_IndexAdd::Params);

  auto self = stack.at(0).toTensor();
  auto dim = stack.at(1).toScalar().to<int>();

  params->axis = get_dim_in_tpc_order(dim, self.dim());
  params->alpha = stack.at(4).toScalar().to<double>();
  return params;
}

OutputMetaDataVector IndexAddMeta(const at::Stack& stack) {
  const auto& input = stack.at(0).toTensor();

  OutputMetaData meta;
  meta.dtype = input.scalar_type();
  meta.shape = input.sizes().vec();
  return {meta};
}

void IndexAdd::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto dtype = ScalarType();
  if (c10::isIntegralType(dtype, true)) {
    update_guid_dtype(guid_, c10::ScalarType::Int);
  }

  size_t size{};
  const auto params = FillIndexAddParams(stack, size);
  const auto meta = IndexAddMeta(stack)[0];

  auto indexAddResult = BuildOp(
      graph,
      guid_,
      {syn_in(0), syn_in(1), syn_in(2)},
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(indexAddResult[0]);
}

} // namespace habana

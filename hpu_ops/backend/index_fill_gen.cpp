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
#include "generated/backend/index_fill.h"

namespace habana {

std::shared_ptr<void> FillIndexFillParams(
    const at::Stack& stack,
    size_t& size) {
  const auto dim =
      at::maybe_wrap_dim(stack.at(1).toInt(), stack.at(0).toTensor().dim());
  PARAMS_STUB(ns_IndexCopy::Params);
  params->axis = dim;
  return params;
}

OutputMetaDataVector IndexFillMeta(const at::Stack& stack) {
  const auto& input = stack.at(0).toTensor();

  OutputMetaData meta;
  meta.dtype = input.scalar_type();
  meta.shape = input.sizes().vec();
  return {meta};
}

void IndexFill::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& input = stack.at(0).toTensor();
  const auto& dim = stack.at(1).toInt();
  const auto& indexes = stack.at(2).toTensor();
  const auto& val = stack.at(3).toScalar().toFloat();

  const auto meta = IndexFillMeta(stack)[0];

  auto valueTensorShape = input.sizes().vec();
  if (!valueTensorShape.empty()) {
    valueTensorShape[dim] = indexes.numel();
  } else {
    HABANA_ASSERT(
        indexes.numel() == 1,
        "For input 0-D tensor, number of elements in indices tensor should be 1.");
  }

  synapse_helpers::tensor valueTensor =
      OpBackend::BuildConstant(this, graph, val, meta.dtype, valueTensorShape);

  size_t size = 0;
  const auto params = FillIndexFillParams(stack, size);

  auto indexCopyResult = BuildOp(
      graph,
      guid_,
      {syn_in(0), syn_in(1), valueTensor.get()},
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(indexCopyResult[0]);
}

} // namespace habana

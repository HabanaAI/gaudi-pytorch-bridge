/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/embedding.h"
#include "generated/backend/embedding_dense_backward.h"

namespace habana {
OutputMetaDataVector EmbeddingMeta(const at::Stack& stack) {
  const auto& weight = stack_tensor(stack, 0);
  const auto& indices = stack_tensor(stack, 1);

  OutputMetaData meta;
  meta.dtype = weight.scalar_type();
  if (indices.dim() == 1) {
    meta.shape = weight.sizes().vec();
    meta.shape[0] = indices.numel();
  } else {
    meta.shape = indices.sizes().vec();
    for (int64_t d : weight.sizes().slice(1)) {
      meta.shape.push_back(d);
    }
  }
  return {meta};
}

std::shared_ptr<void> FillEmbeddingRenormFwdParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_EmbeddingRenormFwdKernel::Params);
  params->max_norm = stack.at(2).toScalar().to<double>();
  params->norm_type = stack.at(3).toScalar().to<double>();
  return params;
}

} // namespace habana

/******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/lazy/embedding.h"
#include "generated/lazy/embedding_dense_backward.h"

namespace habana {
FALLBACK_CHECK(EmbeddingFallbackCheck, bool scale_grad_by_freq, bool sparse) {
  if (scale_grad_by_freq == true || sparse == true) {
    return false;
  } else
    return true;
}
FALLBACK_CHECK(EmbeddingDenseBwdFallbackCheck, bool scale_grad_by_freq) {
  if (scale_grad_by_freq == true) {
    return false;
  } else
    return true;
}

sizes_vec EmbeddingOutputShape(const at::Stack& stack) {
  const auto& weight = stack_tensor(stack, 0);
  const auto& indices = stack_tensor(stack, 1);

  std::vector<int64_t> size;
  if (indices.dim() == 1) {
    size = weight.sizes().vec();
    size[0] = indices.numel();
  } else {
    size = indices.sizes().vec();
    for (int64_t d : weight.sizes().slice(1)) {
      size.push_back(d);
    }
  }
  return {size};
}
} // namespace habana

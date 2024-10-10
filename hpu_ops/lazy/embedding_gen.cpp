/*******************************************************************************
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
} // namespace habana

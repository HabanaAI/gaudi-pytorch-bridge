/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/embedding.h"
#include "generated/embedding_dense_backward.h"

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
} // namespace habana

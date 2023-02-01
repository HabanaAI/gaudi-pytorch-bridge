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
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

FALLBACK_CHECK(IndexSelectFallbackCheck, const at::Tensor& self) {
  if (self.dim() > 5) {
    return false;
  }
  return true;
};
} // namespace habana

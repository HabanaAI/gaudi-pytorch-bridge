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

FALLBACK_CHECK(
    SortStableFallbackCheck,
    const at::Tensor& self,
    c10::optional<bool> stable,
    int64_t dim_,
    bool descending) {
  bool isStable = stable.has_value() ? stable.value() : false;
  auto dim = self.dim() - dim_ - 1;

  if (dim == 0) {
    return !isStable;
  } else if (isStable) {
    if (self.sizes()[dim] <= 37 && descending)
      return true;
    else
      return false;
  } else {
    return true;
  }
}

} // namespace habana

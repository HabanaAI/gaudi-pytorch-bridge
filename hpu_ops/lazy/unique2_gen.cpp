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

#include "generated/lazy/_unique2.h"
namespace habana {
FALLBACK_CHECK(
    Unique2FallbackCheck,
    const at::Tensor& self,
    bool return_inverse,
    bool return_counts) {
  // Fallback as return_inverse & return_counts to true isn't supported
  if (((return_inverse || return_counts)) == true) {
    return false;
  }
  // Fallback as TPC can support only upto 4D(1D to 4D)
  else if (self.dim() > 4) {
    return false;
  } else
    return true;
};
} // namespace habana

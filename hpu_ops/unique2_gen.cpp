/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/_unique2.h"
namespace habana {
FALLBACK_CHECK(
    Unique2FallbackCheck,
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  // Fallback as return_inverse & return_counts to true isn't supported
  if (((return_inverse || return_counts)) == true) {
    return false;
  }
  // Fallback as sorted = True supports only for self with dim 1
  else if (sorted && self.dim() != 1) {
    return false;
  } else
    return true;
};
} // namespace habana
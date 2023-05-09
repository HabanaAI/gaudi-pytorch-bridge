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
#include "generated/backend/trace.h"

namespace habana {

OutputMetaDataVector TraceMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);

  OutputMetaData meta;
  meta.shape = {};
  if (self.scalar_type() == c10::ScalarType::Int) {
    meta.dtype = c10::ScalarType::Long;
  } else {
    meta.dtype = self.scalar_type();
  }
  return {meta};
}

} // namespace habana

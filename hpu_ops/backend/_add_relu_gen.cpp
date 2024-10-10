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

#include "generated/backend/_add_relu.h"

namespace habana {

OutputMetaDataVector AddReluTensorMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto other = stack_tensor(stack, 1);

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = at::infer_size(self.sizes(), other.sizes());
  return {meta};
}

} // namespace habana

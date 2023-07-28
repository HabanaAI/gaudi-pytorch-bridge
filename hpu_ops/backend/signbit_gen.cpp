/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/signbit.h"

namespace habana {

OutputMetaDataVector SignbitMeta(const at::Stack& stack) {
  constexpr size_t SELF_TENSOR_INDEX = 0;

  OutputMetaData meta;
  const at::Tensor& self = stack_tensor(stack, SELF_TENSOR_INDEX);

  meta.shape = self.sizes().vec();
  meta.dtype = at::kBool;

  return {meta};
}

} // namespace habana

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

#include "generated/backend/mul.h"

namespace habana {
OutputMetaDataVector MulMeta(const at::Stack& stack) {
  OutputMetaData meta;
  const at::Tensor self = stack_tensor(stack, 0);
  meta.shape = stack[1].isScalar()
      ? self.sizes().vec()
      : at::infer_size(self.sizes(), stack_tensor(stack, 1).sizes());

  auto m_scalar_type = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      c10::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false,
      c10::nullopt,
      true,
      true);

  meta.dtype = m_scalar_type;
  return {meta};
}
} // namespace habana

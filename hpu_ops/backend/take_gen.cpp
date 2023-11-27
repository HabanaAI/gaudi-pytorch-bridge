/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/take.h"

namespace habana {

OutputMetaDataVector TakeMeta(const at::Stack& stack) {
  const auto input = stack_tensor(stack, 0);
  const auto index = stack_tensor(stack, 1);

  OutputMetaData meta;
  meta.dtype = input.scalar_type();
  meta.shape = index.sizes().vec();
  return {meta};
}
} // namespace habana

/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/vdot.h"

namespace habana {

OutputMetaDataVector VdotMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = {};

  return {meta};
}

} // namespace habana

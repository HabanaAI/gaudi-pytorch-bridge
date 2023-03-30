/******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/clone.h"

namespace habana {

OutputMetaDataVector CloneMeta(const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  OutputMetaData meta;

  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  if (stack.at(1).isNone()) {
    meta.mem_format = self.suggest_memory_format();
  } else {
    meta.mem_format = stack.at(1).toMemoryFormat();
  }

  return {meta};
}

} // namespace habana

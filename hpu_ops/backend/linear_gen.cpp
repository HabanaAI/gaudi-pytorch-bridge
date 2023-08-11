/*
******************************************************************************
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
#include "generated/backend/linear.h"

namespace habana {

OutputMetaDataVector LinearMeta(const at::Stack& stack) {
  const auto& input = stack.at(0).toTensor();
  const auto& weight = stack.at(1).toTensor();
  OutputMetaData meta;
  meta.dtype = input.scalar_type();
  meta.shape = input.sizes().vec();
  meta.shape[input.dim() - 1] = weight.sizes().vec()[0];
  meta.mem_format = input.suggest_memory_format();
  return {meta};
}
} // namespace habana

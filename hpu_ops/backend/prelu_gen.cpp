/******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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
#include <pytorch_helpers/habana_helpers/pt_version_check.h>
#include "generated/backend/_prelu_kernel.h"

namespace habana {
OutputMetaDataVector PreluFwdMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);

  OutputMetaData meta;
  meta.shape = input.sizes().vec();
  meta.dtype = input.scalar_type();

  return {meta};
}

OutputMetaDataVector PreluBwdMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 1);
  const auto& weight = stack_tensor(stack, 2);

  OutputMetaDataVector meta(2);
  meta.at(0).shape = input.sizes().vec();
  meta.at(0).dtype = input.scalar_type();

  meta.at(1).shape = weight.sizes().vec();
  meta.at(1).dtype = weight.scalar_type();
  return meta;
}

} // namespace habana

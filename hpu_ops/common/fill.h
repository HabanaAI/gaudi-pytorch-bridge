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

#pragma once
#include <ATen/core/boxing/BoxedKernel.h>
#include <ATen/core/ivalue_inl.h>

namespace habana {
// NOTE: Workaround due to https://github.com/pytorch/pytorch/issues/75465
inline void CastBoolToInt(at::Stack& stack) {
  auto& maybe_bool_input = stack[1];
  if (maybe_bool_input.isBool()) {
    maybe_bool_input = static_cast<int>(maybe_bool_input.toBool());
  }
}
} // namespace habana

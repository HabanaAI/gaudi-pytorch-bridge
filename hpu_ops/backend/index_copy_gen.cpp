/*******************************************************************************
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
#include "generated/backend/index_copy.h"

namespace habana {

std::shared_ptr<void> FillIndexCopyParams(
    const at::Stack& stack,
    size_t& size) {
  const auto dim = stack[1].toInt();
  PARAMS_STUB(ns_IndexCopy::Params);
  params->axis = dim;
  return params;
}

} // namespace habana

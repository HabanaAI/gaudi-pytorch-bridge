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

#include "generated/backend/log_normal.h"

namespace habana {
std::shared_ptr<void> FillLogNormalParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomNormal::Params);
  params->mean = static_cast<float>(stack.at(1).toDouble());
  params->stddev = static_cast<float>(stack.at(2).toDouble());

  return params;
}
} // namespace habana

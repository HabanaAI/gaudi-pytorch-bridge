/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/floor_divide.h"

namespace habana {
std::shared_ptr<void> FillFloorDivideParams(const at::Stack&, size_t& size) {
  PARAMS_STUB(ns_DivModKernel::ParamsV2);
  // using floor mode
  params->isTruncRoundingMode = false;
  return params;
}
} // namespace habana

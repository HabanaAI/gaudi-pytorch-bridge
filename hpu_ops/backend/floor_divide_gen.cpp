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
#include "habana_kernels/binary_kernels.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

// Except bfloat16, all other types are computed in following type
#define COMMON_COMPUTATION_TYPE_TPC c10::ScalarType::Float

namespace habana {
std::shared_ptr<void> FillFloorDivideParams(const at::Stack&, size_t& size) {
  PARAMS_STUB(ns_DivModKernel::ParamsV2);
  // using floor mode
  params->isTruncRoundingMode = false;
  return params;
}
} // namespace habana

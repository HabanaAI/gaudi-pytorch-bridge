/*******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/isposinf.h"

namespace habana {

static std::shared_ptr<void> FillisposneginfParamsFwd(
    bool detect_positive,
    bool detect_negative,
    size_t& size) {
  PARAMS_STUB(ns_IsInfKernel::Params);
  params->detect_negative = detect_negative;
  params->detect_positive = detect_positive;
  return params;
}

std::shared_ptr<void> FillisinfParamsFwd(const at::Stack&, size_t& size) {
  return FillisposneginfParamsFwd(true, true, size);
}

std::shared_ptr<void> FillisposinfParamsFwd(const at::Stack&, size_t& size) {
  return FillisposneginfParamsFwd(true, false, size);
}

std::shared_ptr<void> FillisneginfParamsFwd(const at::Stack&, size_t& size) {
  return FillisposneginfParamsFwd(false, true, size);
}

} // namespace habana

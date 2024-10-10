/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/hardtanh.h"

namespace habana {
template <typename ScalarType>
static std::shared_ptr<void> HardTanhParams(
    ScalarType min,
    ScalarType max,
    size_t& size) {
  PARAMS_STUB(ns_HardTanhKernel::Params);

  get<ScalarType>(params->lowerBound) = min;
  get<ScalarType>(params->upperBound) = max;

  return params;
}

std::shared_ptr<void> FillHardTanhBwdParams(
    const at::Stack& stack,
    size_t& size) {
  float min = stack[2].isScalar() ? stack[2].toScalar().to<float>()
                                  : -std::numeric_limits<float>::max();
  float max = stack[3].isScalar() ? stack[3].toScalar().to<float>()
                                  : std::numeric_limits<float>::max();
  return HardTanhParams(min, max, size);
}

} // namespace habana

/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"

namespace habana {
template <typename ScalarType>
static std::shared_ptr<void> ClampParams(
    ScalarType min,
    ScalarType max,
    size_t& size) {
  PARAMS_STUB(ns_ClampKernel::Params);

  get<ScalarType>(params->lowerBound) = min;
  get<ScalarType>(params->upperBound) = max;

  return params;
}

std::shared_ptr<void> FillClampParams(const at::Stack& stack, size_t& size) {
  if (c10::isFloatingType(stack[0].toTensor().scalar_type())) {
    float min = stack[1].isScalar() ? stack[1].toScalar().to<float>()
                                    : -std::numeric_limits<float>::max();
    float max = stack[2].isScalar() ? stack[2].toScalar().to<float>()
                                    : std::numeric_limits<float>::max();
    return ClampParams(min, max, size);
  } else {
    int min = stack[1].isScalar() ? stack[1].toScalar().to<int>()
                                  : -std::numeric_limits<int>::max();
    int max = stack[2].isScalar() ? stack[2].toScalar().to<int>()
                                  : std::numeric_limits<int>::max();
    return ClampParams(min, max, size);
  }
}

std::shared_ptr<void> FillClampMinParams(const at::Stack& stack, size_t& size) {
  if (c10::isFloatingType(stack[0].toTensor().scalar_type())) {
    return ClampParams(
        stack[1].toScalar().toFloat(), std::numeric_limits<float>::max(), size);
  }
  return ClampParams(
      stack[1].toScalar().toInt(), std::numeric_limits<int>::max(), size);
}

std::shared_ptr<void> FillClampMaxParams(const at::Stack& stack, size_t& size) {
  if (c10::isFloatingType(stack[0].toTensor().scalar_type())) {
    return ClampParams(
        -std::numeric_limits<float>::max(),
        stack[1].toScalar().toFloat(),
        size);
  }
  return ClampParams(
      -std::numeric_limits<int>::max(), stack[1].toScalar().toInt(), size);
}

} // namespace habana

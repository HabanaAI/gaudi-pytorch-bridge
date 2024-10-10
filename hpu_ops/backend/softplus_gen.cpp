
/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/softplus.h"

namespace habana {
std::shared_ptr<void> FillSoftplusParams(
    const at::Stack& stack,
    size_t& size,
    int beta_index,
    int threshold_index) {
  PARAMS_STUB(ns_Softplus::Params);
  auto beta = stack.at(beta_index).toScalar().to<float>();
  auto threshold = stack.at(threshold_index).toScalar().to<float>();
  params->beta = beta;
  params->threshold = threshold;
  return params;
}
std::shared_ptr<void> FillSoftplusParamsFwd(
    const at::Stack& stack,
    size_t& size) {
  return FillSoftplusParams(
      stack, size, 1 /*beta_index*/, 2 /*threshold_index*/);
}
std::shared_ptr<void> FillSoftplusParamsBwd(
    const at::Stack& stack,
    size_t& size) {
  return FillSoftplusParams(
      stack, size, 2 /*beta_index*/, 3 /*threshold_index*/);
}
} // namespace habana

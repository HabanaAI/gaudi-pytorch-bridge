/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/round.h"

namespace habana {
std::shared_ptr<void> FillRoundParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_RoundKernel::Params);
  static_cast<void>(stack);
  params->roundMode = RoundMode_t::ROUND_HALF_NEAREST_EVEN;
  return params;
}

std::shared_ptr<void> FillRoundDecimalParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RoundKernel::ParamsV2);
  static_cast<void>(stack);
  params->roundMode = RoundMode_t::ROUND_HALF_NEAREST_EVEN;
  params->num_decimal_round = stack.at(1).toScalar().to<int>();
  return params;
}
} // namespace habana

/******************************************************************************
 * Copyright (C) 2021-2023 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/roll.h"

namespace habana {

std::shared_ptr<void> FillRollParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_RollKernel::Params);

  constexpr int64_t shiftIndex = 1;
  constexpr int64_t dimsIndex = 2;

  auto shifts = stack.at(shiftIndex).toIntVector();
  auto dims = stack.at(dimsIndex).toIntVector();

  params->num_dims = dims.size();

  for (size_t i = 0; i < shifts.size(); i++) {
    params->shifts[i] = shifts[i];
  }

  for (size_t i = 0; i < dims.size(); i++) {
    params->dims[i] = dims[i];
  }

  return params;
}

} // namespace habana

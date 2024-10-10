/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/normal.h"

namespace habana {
std::shared_ptr<void> FillNormalParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_RandomNormal::Params);
  params->mean = static_cast<float>(stack.at(1).toDouble());
  params->stddev = static_cast<float>(stack.at(2).toDouble());

  return params;
}
} // namespace habana

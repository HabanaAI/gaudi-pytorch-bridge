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
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
static std::shared_ptr<void> FillRandomNormalParams(
    double mean,
    double std,
    size_t& size) {
  PARAMS_STUB(ns_RandomNormal::Params);
  params->mean = static_cast<float>(mean);
  params->stddev = static_cast<float>(std);

  return params;
}

std::shared_ptr<void> FillNormalParams(const at::Stack& stack, size_t& size) {
  return FillRandomNormalParams(
      stack.at(1).toDouble(), stack.at(2).toDouble(), size);
}
} // namespace habana

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

std::shared_ptr<void> HabanaOperatorHelper::FillHardSigmoidParams(
    const at::Stack&,
    size_t& size) {
  PARAMS_STUB(ns_HardSigmoidKernel::Params);
  constexpr float alpha = 1 / 6.0f;
  constexpr float beta = 1 / 2.0f;

  params->alpha = alpha;
  params->beta = beta;

  return params;
}
} // namespace habana

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
std::shared_ptr<void> HabanaOperatorHelper::FillTriuParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_MatrixBandPartKernel::triParams);
  auto self = stack.at(0).toTensor();
  auto diagonal = stack.at(1).toInt();
  int64_t n = self.sizes()[1];

  params->numLower = diagonal;
  params->numUpper = n;
  params->excludeDiag = 1;
  return params;
}

std::shared_ptr<void> HabanaOperatorHelper::FillTrilParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_MatrixBandPartKernel::triParams);
  auto self = stack.at(0).toTensor();
  auto diagonal = stack.at(1).toInt();
  int64_t m = self.sizes()[0];

  params->numLower = -m;
  params->numUpper = diagonal;
  params->excludeDiag = 1;
  return params;
}

} // namespace habana

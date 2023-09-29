/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/tril.h"
#include "generated/backend/triu.h"

namespace habana {
std::shared_ptr<void> FillTriuParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_MatrixBandPartKernel::triParams);
  auto self = stack.at(0).toTensor();
  auto diagonal = stack.at(1).toInt();

  params->numLower = diagonal;
  params->numUpper = INT_MAX;
  params->excludeDiag = 1;
  return params;
}

std::shared_ptr<void> FillTrilParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_MatrixBandPartKernel::triParams);
  auto self = stack.at(0).toTensor();
  auto diagonal = stack.at(1).toInt();

  params->numLower = INT_MIN;
  params->numUpper = diagonal;
  params->excludeDiag = 1;
  return params;
}

} // namespace habana

/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/tril.h"
#include "generated/triu.h"

namespace habana {
std::shared_ptr<void> FillTriuParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_MatrixBandPartKernel::triParams);
  auto self = stack.at(0).toTensor();
  auto diagonal = stack.at(1).toInt();
  int dim = self.ndimension();
  int64_t n = self.sizes()[dim - 1];

  params->numLower = diagonal;
  params->numUpper = n - 1;
  params->excludeDiag = 1;
  return params;
}

std::shared_ptr<void> FillTrilParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_MatrixBandPartKernel::triParams);
  auto self = stack.at(0).toTensor();
  auto diagonal = stack.at(1).toInt();
  int dim = self.ndimension();
  int64_t m = self.sizes()[dim - 2];

  params->numLower = -(m - 1);
  params->numUpper = diagonal;
  params->excludeDiag = 1;
  return params;
}

} // namespace habana

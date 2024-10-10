/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "generated/backend/sigmoid.h"

namespace habana {

std::shared_ptr<void> FillSigmoidParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_SigmoidKernel::Params);
  const auto input_dtype = stack[0].toTensor().scalar_type();
  if (input_dtype == at::ScalarType::Float or
      input_dtype == at::ScalarType::BFloat16) {
    params->flavor = NO_SATURATION_SIGMOID;
  } else {
    params->flavor = SIGMOID_DEFAULT;
  }
  return params;
}

} // namespace habana

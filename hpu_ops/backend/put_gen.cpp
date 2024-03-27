
/*******************************************************************************
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
#include <perf_lib_layer_params.h>
#include "generated/backend/put.h"

namespace habana {

std::shared_ptr<void> FillPutParams(const at::Stack& stack, size_t& size) {
  const bool accumulate = stack.at(3).toBool();
  PARAMS_STUB(ns_PutKernel::Params);
  params->accumulate = accumulate;
  return params;
}
} // namespace habana

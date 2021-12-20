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
std::shared_ptr<void> FillLeakyReluBackwardParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_LeakyReluKernel::Params);
  auto alpha = stack.at(2).toScalar().to<float>();
  bool is_result = stack.at(3).toBool();
  TORCH_CHECK(
      !is_result || alpha >= 0.0,
      "In-place leakyReLu backward calculation is triggered with a negative slope which is not supported. "
      "This is caused by calling in-place forward function with a negative slope, "
      "please call out-of-place version instead. File an issue at https://github.com/pytorch/pytorch if you do "
      "require supporting in-place leakRelu backward calculation with negative slope");
  params->alpha = alpha;
  return params;
}
} // namespace habana

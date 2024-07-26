/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/elu.h"
#include "generated/backend/elu_backward.h"

namespace habana {
std::shared_ptr<void> FillEluParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_EluKernel::Params);
  params->alpha = stack.at(1).toScalar().toFloat();
  return params;
}

std::shared_ptr<void> FillEluBackwardParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_EluKernel::ParamsV2);
  float alpha = stack.at(1).toScalar().to<float>();
  float scale = stack.at(2).toScalar().to<float>();
  float input_scale = stack.at(3).toScalar().to<float>();
  bool is_result = stack.at(4).toBool();
  TORCH_CHECK(scale == 1.0, "scale = 1 is only supported");
  TORCH_CHECK(input_scale == 1.0, "input_scale = 1 is only supported");
  TORCH_CHECK(is_result == false, "is_result = false is only supported");
  params->alpha = alpha;
  params->isInputFeaturemap = true;
  return params;
}

} // namespace habana

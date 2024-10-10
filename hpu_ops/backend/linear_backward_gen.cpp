/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/linear_backward.h"

namespace habana {
OutputMetaDataVector LinearBackwardMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);
  const auto& weight = stack_tensor(stack, 2);
  const auto& grad_mask = stack.at(3).toBoolList();
  std::vector<int64_t> bias_grad_shape;
  if (grad_mask[2]) {
    bias_grad_shape.push_back(weight.sizes().vec()[0]);
  } else {
    bias_grad_shape.push_back(1);
  }
  OutputMetaData input_meta, weight_meta, bias_meta;

  input_meta.shape = input.sizes().vec();
  input_meta.dtype = input.scalar_type();

  weight_meta.shape = weight.sizes().vec();
  weight_meta.dtype = weight.scalar_type();

  bias_meta.shape = bias_grad_shape;
  bias_meta.dtype = weight.scalar_type();

  return {input_meta, weight_meta, bias_meta};
}

std::shared_ptr<void> FillLinearBwdParams(
    const at::Stack& stack,
    size_t& size) {
  const auto& grad_mask = stack.at(3).toBoolList();
  PARAMS_STUB(ns_LinearBwdKernel::Params);
  params->gradBias = grad_mask[2];

  return params;
}

} // namespace habana
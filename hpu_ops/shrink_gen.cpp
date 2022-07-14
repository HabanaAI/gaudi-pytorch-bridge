/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/hardshrink.h"
#include "generated/softshrink.h"
#include "hpu_op_helper.h"

namespace habana {
// mode_t = softshrink/hardshrink
// index_lambda  = index position of lambda
static std::shared_ptr<void> FillshrinkParams(
    const at::Stack& stack,
    size_t& size,
    ShrinkMode_t mode_t,
    int index_lambda) {
  PARAMS_STUB(ns_ShrinkKernel::TrainingParams);
  float lambda = stack.at(index_lambda).toScalar().to<float>();
  params->lowerBound = -lambda;
  params->upperBound = lambda;
  params->mode = mode_t;
  return params;
}

std::shared_ptr<void> FillhardshrinkfwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillshrinkParams(stack, size, ShrinkMode_t::HARD_SHRINK, 1);
}

std::shared_ptr<void> FillsoftshrinkfwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillshrinkParams(stack, size, ShrinkMode_t::SOFT_SHRINK, 1);
}

std::shared_ptr<void> FillhardshrinkbwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillshrinkParams(stack, size, ShrinkMode_t::HARD_SHRINK, 2);
}

std::shared_ptr<void> FillsoftshrinkbwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillshrinkParams(stack, size, ShrinkMode_t::SOFT_SHRINK, 2);
}
} // namespace habana

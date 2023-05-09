/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/logit.h"

namespace habana {
std::shared_ptr<void> FillLogitParams(
    const at::Stack& stack,
    size_t& size,
    int64_t index) {
  // check if eps=None
  if (stack.at(index).isNone())
    return nullptr;

  PARAMS_STUB(ns_LogitKernel::Params);
  params->epsilon = stack.at(index).toDouble();

  return params;
}

std::shared_ptr<void> FillLogitForwardParams(
    const at::Stack& stack,
    size_t& size) {
  // index positions for input args
  constexpr size_t epsPositionInArgList = 1;

  return FillLogitParams(stack, size, epsPositionInArgList);
}

std::shared_ptr<void> FillLogitBackwardParams(
    const at::Stack& stack,
    size_t& size) {
  // index positions for input args
  constexpr size_t epsPositionInArgList = 2;

  return FillLogitParams(stack, size, epsPositionInArgList);
}
} // namespace habana

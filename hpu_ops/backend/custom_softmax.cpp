/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/custom_softmax.h"

namespace habana {

std::shared_ptr<void> FillCustomSoftmaxParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_CustomSoftmax::Params);
  params->flavor = stack[1].toInt();
  return params;
}

CustomSoftmax::CustomSoftmax(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "custom_softmax_fwd",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetFillParams(FillCustomSoftmaxParams);
}

} // namespace habana

static const auto& CustomSoftmaxKernelRegistry = habana::KernelRegistry().add(
    "hpu::custom_softmax",
    KERNEL_FN_GLOBAL(habana::CustomSoftmax));

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

#include "hpu_ops/scaled_masked_softmax.h"

namespace habana {

std::shared_ptr<void> FillScaledMaskedSoftmaxParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_SmoothL1Kernel::Params);
  params->sigma = stack[2].toDouble();
  return params;
}

ScaledMaskedSoftmax::ScaledMaskedSoftmax(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "scaled_masked_softmax_fwd_",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetFillParams(FillScaledMaskedSoftmaxParams);
}

} // namespace habana

static const auto& ScaledMaskedSoftmaxKernelRegistry =
    habana::KernelRegistry().add(
        "hpu::scaled_masked_softmax",
        KERNEL_FN_GLOBAL(habana::ScaledMaskedSoftmax));

/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
#pragma once
#include <ATen/core/ATen_fwd.h>
#include <ATen/native/ReduceOpsUtils.h>
#include "habana_helpers/logging.h"
#include "hpu_ops/op_backend.h"

namespace habana {

sizes_vec ReductionOutputShape(
    const at::Tensor& self,
    at::OptionalIntArrayRef dims,
    bool keepdim);

unsigned ReductionMask(const at::Tensor& self, at::optional<int64_t> dimOpt);

at::optional<at::ScalarType> get_dtype(
    const at::Stack& stack,
    at::optional<uint8_t> dtype_index);

std::vector<int64_t> get_dims(
    const at::Stack& stack,
    at::optional<uint8_t> dim_index);

inline bool get_keepdim(
    const at::Stack& stack,
    at::optional<uint8_t> keepdim_index) {
  return keepdim_index.has_value() ? stack.at(keepdim_index.value()).toBool()
                                   : false;
}

inline std::pair<unsigned, int>
getMaskWithBitPosOutInTpcOrderAndBitPosInTpcOrder(int bitPos, int ndims) {
  int bitPosInTpcOrder = ndims - 1 - bitPos;
  unsigned fullMask = (1 << ndims) - 1;

  unsigned maskBitPosInTpcOrder =
      (bitPosInTpcOrder >= 0) ? 1 << bitPosInTpcOrder : 0;
  unsigned maskedOutBitPos = fullMask & ~maskBitPosInTpcOrder;

  return {maskedOutBitPos, bitPosInTpcOrder};
}

inline unsigned getMaskWithBitPosOutInTpcOrder(int bitPos, int ndims) {
  return getMaskWithBitPosOutInTpcOrderAndBitPosInTpcOrder(bitPos, ndims).first;
}

} // namespace habana

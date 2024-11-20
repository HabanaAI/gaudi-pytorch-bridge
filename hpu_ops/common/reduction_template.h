/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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

/**
 * Copyright (c) 2023-2024 Intel Corporation
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

#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_backend.h"

namespace habana {

ns_CastKernel::Params GetCastParams(
    const bool stochastic,
    const at::ScalarType& from_dtype,
    const at::ScalarType& to_dtype);

OutputMetaDataVector CastToFp8V2Meta(const at::Stack& stack);

// Determines if STOCHASTIC_FLUSH_TO_ZERO should be used instead
// of STORCHASTIC_ROUNDING.
const bool is_sr_sftz = GET_ENV_FLAG_NEW(PT_HPU_STOCHASTIC_ROUNDING_MODE) == 1;

inline SharedMetaTensor getSharedMetaTensorFromScale(const at::IValue& scale) {
  if (scale.isNone()) {
    return createOptionalNotPresentSharedMetaTensor();
  }
  if (scale.isTensor()) {
    return getSharedMetaFromTensor(scale.toTensor());
  }
  return {1, at::ScalarType::Float};
}

} // namespace habana

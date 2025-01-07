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

#include "generated/backend/hardtanh.h"

namespace habana {
template <typename ScalarType>
static std::shared_ptr<void> HardTanhParams(
    ScalarType min,
    ScalarType max,
    size_t& size) {
  PARAMS_STUB(ns_HardTanhKernel::Params);

  get<ScalarType>(params->lowerBound) = min;
  get<ScalarType>(params->upperBound) = max;

  return params;
}

std::shared_ptr<void> FillHardTanhBwdParams(
    const at::Stack& stack,
    size_t& size) {
  float min = stack[2].isScalar() ? stack[2].toScalar().to<float>()
                                  : -std::numeric_limits<float>::max();
  float max = stack[3].isScalar() ? stack[3].toScalar().to<float>()
                                  : std::numeric_limits<float>::max();
  return HardTanhParams(min, max, size);
}

} // namespace habana

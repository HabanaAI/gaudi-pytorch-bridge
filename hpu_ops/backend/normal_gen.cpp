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

#include "generated/backend/normal.h"

namespace habana {
std::shared_ptr<void> FillNormalParams(const at::Stack& stack, size_t& size) {
  static const bool use_philox = GET_ENV_FLAG_NEW(PT_HPU_USE_PHILOX_NORMAL);
  PARAMS_STUB(ns_RandomNormal::ParamsV2);
  params->mean = static_cast<float>(stack.at(1).toDouble());
  params->stddev = static_cast<float>(stack.at(2).toDouble());
  params->usePhilox = use_philox;

  return params;
}
} // namespace habana

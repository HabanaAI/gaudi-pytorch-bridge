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

#include "generated/backend/round.h"

namespace habana {
std::shared_ptr<void> FillRoundParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_RoundKernel::Params);
  static_cast<void>(stack);
  params->roundMode = RoundMode_t::ROUND_HALF_NEAREST_EVEN;
  return params;
}

std::shared_ptr<void> FillRoundDecimalParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RoundKernel::ParamsV2);
  static_cast<void>(stack);
  params->roundMode = RoundMode_t::ROUND_HALF_NEAREST_EVEN;
  params->num_decimal_round = stack.at(1).toScalar().to<int>();
  return params;
}
} // namespace habana

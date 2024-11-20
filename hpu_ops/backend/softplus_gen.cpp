
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
#include "generated/backend/softplus.h"

namespace habana {
std::shared_ptr<void> FillSoftplusParams(
    const at::Stack& stack,
    size_t& size,
    int beta_index,
    int threshold_index) {
  PARAMS_STUB(ns_Softplus::Params);
  auto beta = stack.at(beta_index).toScalar().to<float>();
  auto threshold = stack.at(threshold_index).toScalar().to<float>();
  params->beta = beta;
  params->threshold = threshold;
  return params;
}
std::shared_ptr<void> FillSoftplusParamsFwd(
    const at::Stack& stack,
    size_t& size) {
  return FillSoftplusParams(
      stack, size, 1 /*beta_index*/, 2 /*threshold_index*/);
}
std::shared_ptr<void> FillSoftplusParamsBwd(
    const at::Stack& stack,
    size_t& size) {
  return FillSoftplusParams(
      stack, size, 2 /*beta_index*/, 3 /*threshold_index*/);
}
} // namespace habana

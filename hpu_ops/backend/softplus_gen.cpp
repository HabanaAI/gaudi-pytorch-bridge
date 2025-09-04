/**
 * Copyright (c) 2021-2025 Intel Corporation
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
FillParamsT FillSoftplusParams(
    const at::Stack& stack,
    int beta_index,
    int threshold_index) {
  PARAMS_STUB(ns_Softplus::Params);
  auto beta = stack.at(beta_index).toScalar().to<float>();
  auto threshold = stack.at(threshold_index).toScalar().to<float>();
  params->beta = beta;
  params->threshold = threshold;
  return paramsT;
}
FillParamsT FillSoftplusParamsFwd(const at::Stack& stack) {
  return FillSoftplusParams(stack, 1 /*beta_index*/, 2 /*threshold_index*/);
}
FillParamsT FillSoftplusParamsBwd(const at::Stack& stack) {
  return FillSoftplusParams(stack, 2 /*beta_index*/, 3 /*threshold_index*/);
}
} // namespace habana

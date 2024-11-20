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

#include "generated/backend/isposinf.h"

namespace habana {

static std::shared_ptr<void> FillisposneginfParamsFwd(
    bool detect_positive,
    bool detect_negative,
    size_t& size) {
  PARAMS_STUB(ns_IsInfKernel::Params);
  params->detect_negative = detect_negative;
  params->detect_positive = detect_positive;
  return params;
}

std::shared_ptr<void> FillisinfParamsFwd(const at::Stack&, size_t& size) {
  return FillisposneginfParamsFwd(true, true, size);
}

std::shared_ptr<void> FillisposinfParamsFwd(const at::Stack&, size_t& size) {
  return FillisposneginfParamsFwd(true, false, size);
}

std::shared_ptr<void> FillisneginfParamsFwd(const at::Stack&, size_t& size) {
  return FillisposneginfParamsFwd(false, true, size);
}

} // namespace habana

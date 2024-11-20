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
#include "generated/backend/sigmoid.h"

namespace habana {

std::shared_ptr<void> FillSigmoidParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_SigmoidKernel::Params);
  const auto input_dtype = stack[0].toTensor().scalar_type();
  if (input_dtype == at::ScalarType::Float or
      input_dtype == at::ScalarType::BFloat16) {
    params->flavor = NO_SATURATION_SIGMOID;
  } else {
    params->flavor = SIGMOID_DEFAULT;
  }
  return params;
}

} // namespace habana

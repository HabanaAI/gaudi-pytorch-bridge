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

#include "generated/backend/leaky_relu.h"
#include "generated/backend/leaky_relu_backward.h"

namespace habana {
std::shared_ptr<void> FillLeakyReluParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_LeakyReluKernel::Params);
  params->alpha = stack.at(1).toScalar().toFloat();
  return params;
}

std::shared_ptr<void> FillLeakyReluBackwardParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_LeakyReluKernel::Params);
  auto alpha = stack.at(2).toScalar().to<float>();
  bool is_result = stack.at(3).toBool();
  TORCH_CHECK(
      !is_result || alpha >= 0.0,
      "In-place leakyReLu backward calculation is triggered with a negative slope which is not supported. "
      "This is caused by calling in-place forward function with a negative slope, "
      "please call out-of-place version instead. File an issue at https://github.com/pytorch/pytorch if you do "
      "require supporting in-place leakRelu backward calculation with negative slope");
  params->alpha = alpha;
  return params;
}
} // namespace habana

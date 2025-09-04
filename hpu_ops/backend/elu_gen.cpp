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

#include "generated/backend/elu.h"
#include "generated/backend/elu_backward.h"

namespace habana {
FillParamsT FillEluParams(const at::Stack& stack) {
  PARAMS_STUB(ns_EluKernel::Params);
  params->alpha = stack.at(1).toScalar().toFloat();
  return paramsT;
}

FillParamsT FillEluBackwardParams(const at::Stack& stack) {
  PARAMS_STUB(ns_EluKernel::ParamsV2);
  float alpha = stack.at(1).toScalar().to<float>();
  float scale = stack.at(2).toScalar().to<float>();
  float input_scale = stack.at(3).toScalar().to<float>();
  bool is_result = stack.at(4).toBool();
  HABANA_ASSERT(scale == 1.0, "scale = 1 is only supported");
  HABANA_ASSERT(input_scale == 1.0, "input_scale = 1 is only supported");
  HABANA_ASSERT(is_result == false, "is_result = false is only supported");
  params->alpha = alpha;
  params->isInputFeaturemap = true;
  return paramsT;
}

} // namespace habana

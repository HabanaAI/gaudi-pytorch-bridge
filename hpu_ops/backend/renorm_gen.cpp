/**
 * Copyright (c) 2025 Intel Corporation
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

#include "generated/backend/renorm.h"

namespace habana {
FillParamsT FillRenormParams(const at::Stack& stack) {
  PARAMS_STUB(ns_RenormKernel::Params);
  params->p = stack.at(1).toScalar().to<double>();
  params->dim = stack.at(2).toInt();
  params->max_norm = stack.at(3).toScalar().to<double>();

  return paramsT;
}

} // namespace habana

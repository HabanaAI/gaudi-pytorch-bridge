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

#include "generated/backend/ind2ptr.h"

namespace habana {

FillParamsT FillInd2ptrParams(const at::Stack& stack) {
  PARAMS_STUB(ns_Ind2ptr::Params);
  params->size = stack.at(1).toInt();

  return paramsT;
}

OutputMetaDataVector Ind2ptrMeta(const at::Stack& stack) {
  const at::Tensor& input = stack.at(0).toTensor();
  int64_t M = stack.at(1).toInt();

  return {{input.scalar_type(), {M + 1}}};
}

} // namespace habana

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

#include "generated/backend/eye.h"
#include "habana_helpers/conversion.h"

namespace habana {
OutputMetaDataVector EyeMeta(const at::Stack& stack) {
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  const int64_t n = stack.at(0).toInt();
  if (stack.size() == 3) {
    const int64_t m = stack.at(1).toInt();
    meta.dtype = stack_tensor(stack, 2).scalar_type();
    meta.shape = {n, m};
  } else {
    meta.dtype = stack_tensor(stack, 1).scalar_type();
    meta.shape = {n, n};
  }
  return metaVec;
}

FillParamsT FillEyeParams(const at::Stack& stack) {
  PARAMS_STUB(ns_Eye::Params);

  const auto n = safe_convert<int>(stack.at(0).toInt());
  params->rows = n;
  if (stack.size() == 3) {
    const auto m = safe_convert<int>(stack.at(1).toInt());
    params->cols = m;
  } else {
    params->cols = n;
  }
  return paramsT;
}
} // namespace habana

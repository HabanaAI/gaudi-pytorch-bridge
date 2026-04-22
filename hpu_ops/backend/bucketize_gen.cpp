/**
 * Copyright (c) 2021-2026 Intel Corporation
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

#include "generated/backend/bucketize.h"

namespace habana {

FillParamsT FillBucketizeParams(const at::Stack& stack) {
  PARAMS_STUB(ns_SearchSorted::Params);
  params->right = static_cast<int>(stack.at(3).toBool());
  return paramsT;
}

OutputMetaDataVector BucketizeMeta(const at::Stack& stack) {
  bool out_int32 = stack.at(2).toBool();

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  if (stack.at(0).isTensor()) {
    meta.shape = stack_tensor(stack, 0).sizes().vec();
  } else {
    meta.shape = {1};
  }
  meta.dtype = out_int32 ? torch::kInt32 : torch::kLong;
  return metaVec;
}

} // namespace habana

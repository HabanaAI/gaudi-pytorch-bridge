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

#include "generated/backend/bucketize.h"

namespace habana {

std::shared_ptr<void> FillBucketizeParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_SearchSorted::Params);
  params->right = stack.at(3).toBool();
  return params;
}

OutputMetaDataVector BucketizeMeta(const at::Stack& stack) {
  std::vector<int64_t> outshape;
  if (stack.at(0).isTensor()) {
    outshape = stack_tensor(stack, 0).sizes().vec();
  } else {
    outshape = {1};
  }
  bool out_int32 = stack.at(2).toBool();

  OutputMetaData meta;
  meta.shape = outshape;
  meta.dtype = out_int32 ? torch::kInt32 : torch::kLong;
  return {meta};
}

} // namespace habana

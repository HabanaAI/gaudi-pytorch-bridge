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
#include "generated/backend/equal.h"

namespace habana {

std::shared_ptr<void> FillEqualParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_EqualPt::Params);
  auto self_sizes = stack_tensor(stack, 0).sizes();
  auto other_sizes = stack_tensor(stack, 1).sizes();
  params->forceFalse = self_sizes.size() != other_sizes.size();
  return params;
}

OutputMetaDataVector EqualMeta(const at::Stack&) {
  OutputMetaData meta;
  meta.shape = {};
  meta.dtype = c10::ScalarType::Bool;
  return {meta};
}

} // namespace habana

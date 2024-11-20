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

#include "generated/backend/eq.h"
#include "generated/backend/ge.h"
#include "generated/backend/gt.h"
#include "generated/backend/le.h"
#include "generated/backend/lt.h"
#include "generated/backend/ne.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {
OutputMetaDataVector CompareMeta(const at::Stack& stack) {
  OutputMetaData meta;
  const at::Tensor self = stack_tensor(stack, 0);
  meta.shape = stack[1].isScalar()
      ? self.sizes().vec()
      : at::infer_size(self.sizes(), stack_tensor(stack, 1).sizes());
  meta.dtype = at::kBool;
  return {meta};
}

SharedMetaDataVector CompareEqSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return CompareSharedMeta(stack, "equal_fwd");
}

SharedMetaDataVector CompareGeSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return CompareSharedMeta(stack, "greater_equal_fwd");
}

SharedMetaDataVector CompareGtSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return CompareSharedMeta(stack, "greater_fwd");
}

SharedMetaDataVector CompareLeSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return CompareSharedMeta(stack, "less_equal_fwd");
}

SharedMetaDataVector CompareLtSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return CompareSharedMeta(stack, "less_fwd");
}

} // namespace habana

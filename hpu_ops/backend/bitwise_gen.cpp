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
#include <cstdint>
#include "generated/backend/bitwise_and.h"
#include "generated/backend/bitwise_or.h"
#include "generated/backend/bitwise_xor.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {

std::vector<int64_t> BitwiseLogicalShape(const at::Stack& stack) {
  if (stack.at(0).isScalar() && stack.at(1).isTensor()) {
    return {stack_tensor(stack, 1).sizes().vec()};
  }
  const torch::Tensor& self = stack_tensor(stack, 0);
  if (stack.at(1).isScalar()) {
    return {self.sizes().vec()};
  }
  const torch::Tensor& other = stack_tensor(stack, 1);
  return at::infer_size(self.sizes(), other.sizes());
}

OutputMetaDataVector BitwiseLogicalMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = BitwiseLogicalShape(stack);
  meta.dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      {stack[0], stack[1]},
      c10::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false);
  return {meta};
}

SharedMetaDataVector BitwiseAndSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return BitwiseLogicalSharedMeta(stack, "bitwise_and_fwd");
}

SharedMetaDataVector BitwiseOrSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return BitwiseLogicalSharedMeta(stack, "bitwise_or_fwd");
}

SharedMetaDataVector BitwiseXorSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return BitwiseLogicalSharedMeta(stack, "bitwise_xor_fwd");
}

SharedMetaDataVector BitwiseShiftSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return BitwiseLogicalSharedMeta(stack, "bitshift_fwd");
}

} // namespace habana

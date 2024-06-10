/******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include <cstdint>
#include "generated/backend/bitwise_and.h"
#include "generated/backend/bitwise_or.h"
#include "generated/backend/bitwise_xor.h"

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

} // namespace habana

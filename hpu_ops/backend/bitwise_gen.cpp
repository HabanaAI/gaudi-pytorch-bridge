/******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

OutputMetaDataVector BitwiseLogicalMetaCommon(
    const at::Stack& stack,
    const at::ScalarType& dtype) {
  OutputMetaData meta;
  meta.shape = BitwiseLogicalShape(stack);
  meta.dtype = dtype;
  return {meta};
}

OutputMetaDataVector BitwiseLogicalMeta(const at::Stack& stack) {
  const auto dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      c10::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false);
  return BitwiseLogicalMetaCommon(stack, dtype);
}

OutputMetaDataVector BitwiseLogicalMetaOut(const at::Stack& stack) {
  const auto dtype = stack.back().toTensor().scalar_type();
  return BitwiseLogicalMetaCommon(stack, dtype);
}

} // namespace habana

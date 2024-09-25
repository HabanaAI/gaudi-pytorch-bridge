/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/index_copy.h"

namespace habana {

OutputMetaDataVector IndexCopyMeta(const at::Stack& stack) {
  const auto& input_tensor = stack.at(0).toTensor();
  const auto& dim = stack.at(1).toInt();
  const auto& copy_tensor = stack.at(3).toTensor();
  auto inputTensorShape = input_tensor.sizes().vec();
  auto copyTensorShape = copy_tensor.sizes().vec();
  inputTensorShape.erase(inputTensorShape.begin() + dim);
  copyTensorShape.erase(copyTensorShape.begin() + dim);
  TORCH_CHECK(
      inputTensorShape == copyTensorShape,
      " Source/destination tensor must have same slice shapes. Destination slice shape: ",
      inputTensorShape,
      " at dimension ",
      dim,
      " and source slice shape: ",
      copyTensorShape,
      " at dimension ",
      dim);
  OutputMetaData meta;
  meta.dtype = input_tensor.scalar_type();
  meta.shape = input_tensor.sizes().vec();
  return {meta};
}

std::shared_ptr<void> FillIndexCopyParams(
    const at::Stack& stack,
    size_t& size) {
  const auto dim = stack[1].toInt();
  PARAMS_STUB(ns_IndexCopy::Params);
  params->axis = dim;
  return params;
}

} // namespace habana

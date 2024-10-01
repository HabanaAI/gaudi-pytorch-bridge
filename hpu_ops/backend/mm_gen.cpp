/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/mm.h"
namespace habana {
OutputMetaDataVector MmMeta(const at::Stack& stack) {
  TORCH_CHECK(
      (stack.at(0).isTensor() && stack.at(1).isTensor()),
      " Matmul Input type expected to be tensors");
  auto mat1 = stack.at(0).toTensor();
  auto mat2 = stack.at(1).toTensor();
  TORCH_CHECK(
      mat1.scalar_type() == mat2.scalar_type(),
      "expected m1 and m2 to have the same dtype, but got: ",
      mat1.scalar_type(),
      " != ",
      mat2.scalar_type());

  OutputMetaData meta;
  meta.shape = {mat1.size(0), mat2.size(1)};
  meta.dtype = mat1.scalar_type();
  return {meta};
}
} // namespace habana

/*
******************************************************************************
* Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/linear.h"

namespace habana {

OutputMetaDataVector LinearMeta(const at::Stack& stack) {
  const auto& input = stack.at(0).toTensor();
  const auto& weight = stack.at(1).toTensor();
  OutputMetaData meta;
  meta.dtype = input.scalar_type();
  meta.shape = input.sizes().vec();
  meta.shape[input.dim() - 1] = weight.sizes().vec()[0];
  // Condition check to detect input with incompatible shapes
  // Number of dimensions in matrix 1 can vary
  int mat1_dim0 = 1, dim_i = 0;
  for (; dim_i < input.dim() - 1; ++dim_i)
      mat1_dim0 *= input.sizes().vec()[dim_i];
  TORCH_CHECK(
      input.sizes().vec()[input.dim()-1] == weight.sizes().vec()[1], "matrix 1 and matrix 2 shapes cannot be multiplied (",
      mat1_dim0, "x", input.sizes().vec()[input.dim()-1], " and ",
      weight.sizes().vec()[1], "x", weight.sizes().vec()[0], ")");

  return {meta};
}
} // namespace habana

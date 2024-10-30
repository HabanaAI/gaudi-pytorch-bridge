/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/sum_fp8.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {

OutputMetaDataVector SumFp8Meta(const at::Stack& stack) {
  auto input = stack_tensor(stack, 0);
  auto dims = get_dims(stack, 1);
  auto keepdims = stack[2].toBool();
  auto out_dtype =
      stack[3].toOptional<at::ScalarType>().value_or(input.scalar_type());
  at::DimVector shape =
      at::meta::get_reduction_shape(stack_tensor(stack, 0), dims, keepdims);

  OutputMetaData meta;
  meta.dtype = out_dtype;
  meta.shape = std::vector<int64_t>(shape.begin(), shape.end());
  return {meta};
}

std::shared_ptr<void> FillSumFp8Params(const at::Stack& stack, size_t& size) {
  auto ndims = stack_tensor(stack, 0).dim();
  auto dims = get_dims(stack, 1);
  auto keepdim = stack[2].toBool();

  PARAMS_STUB(ns_Reduction::ParamsV2);
  unsigned maskval = 0;
  for (size_t i = 0; i < dims.size(); ++i) {
    auto d = c10::maybe_wrap_dim(dims[i], ndims); // handling negative indices
    maskval = maskval | (1 << (ndims - d - 1)); // (ndims-i-1) is TPC order
  }
  params->reductionDimensionMask = maskval;
  params->keepDim = keepdim;
  return params;
}

} // namespace habana

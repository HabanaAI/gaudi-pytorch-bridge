/*******************************************************************************
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
#include "generated/backend/count_nonzero.h"

namespace habana {

static std::vector<int64_t> get_dims_from_stack(const at::Stack& stack) {
  auto tensor_rank = stack_tensor(stack, 0).sizes().size();

  std::vector<int64_t> dims;
  if (stack[1].isIntList()) {
    dims = stack[1].toIntList().vec();
  } else if (stack[1].isInt()) {
    dims = {stack[1].toInt()};
  }
  for (int64_t& dim : dims) {
    if (dim < 0) {
      dim = tensor_rank + dim;
    }
  }

  return dims;
}

std::shared_ptr<void> FillCountNonzeroParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_CountNonZero::Params);
  params->dims = 0;

  std::vector<int64_t> dims = get_dims_from_stack(stack);
  if (dims.empty()) {
    params->dims = (1 << stack_tensor(stack, 0).sizes().size()) - 1;
  } else {
    for (int dim : dims) {
      params->dims |= (1 << dim);
    }
  }

  return params;
}

OutputMetaDataVector CountNonzeroMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto self_shape = self.sizes();

  OutputMetaData meta;

  meta.dtype = c10::ScalarType::Long;

  std::vector<int64_t> dims = get_dims_from_stack(stack);
  std::vector<int64_t> output_shape = {};

  if (!dims.empty()) {
    for (uint64_t i = 0; i < self_shape.size(); ++i) {
      if (std::find(dims.begin(), dims.end(), i) == dims.end()) {
        output_shape.push_back(self_shape[i]);
      }
    }
  }

  meta.shape = output_shape;

  return {meta};
}

} // namespace habana

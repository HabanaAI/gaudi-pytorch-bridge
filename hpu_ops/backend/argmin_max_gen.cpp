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

#include "common/utils.h"
#include "generated/backend/argmax.h"
#include "generated/backend/argmin.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {

OutputMetaDataVector ArgMinMaxMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const auto dimOpt = stack.at(1);
  const bool keepdim = stack.at(2).toBool();

  auto dimVector = dimOpt.isNone() ? std::vector<int64_t>{}
                                   : std::vector<int64_t>{dimOpt.toInt()};

  OutputMetaData meta;
  meta.shape = ReductionOutputShape(self, dimVector, keepdim)[0];
  meta.dtype = c10::ScalarType::Long;
  return {meta};
}

std::shared_ptr<void> FillArgMinMaxParams(
    const at::Stack& stack,
    size_t& size) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dimOpt = stack.at(1).toOptional<int64_t>();

  PARAMS_STUB(ns_Reduction::ParamsV2);
  params->reductionDimensionMask = ReductionMask(self, dimOpt);
  params->keepDim = stack.at(2).toBool();
  return params;
}

} // namespace habana

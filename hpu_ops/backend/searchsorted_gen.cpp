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
#include "generated/backend/searchsorted.h"

namespace habana {
OutputMetaDataVector SearchSortedMeta(const at::Stack& stack) {
  std::vector<int64_t> outshape;
  if (stack.at(1).isTensor()) {
    auto self = stack_tensor(stack, 1);
    outshape = self.sizes().vec();
  } else {
    outshape = {1};
  }
  bool out_int32 = stack.at(2).toBool();

  OutputMetaData meta;
  meta.shape = outshape;
  meta.dtype = out_int32 ? torch::kInt32 : torch::kLong;
  return {meta};
}

std::shared_ptr<void> FillSearchSortedParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_SearchSorted::Params);
  bool right = stack.at(3).toBool();
  if (stack.at(4).isString()) {
    right = stack.at(4).toStringView() == "right";
  }

  params->right = right;
  return params;
}

} // namespace habana
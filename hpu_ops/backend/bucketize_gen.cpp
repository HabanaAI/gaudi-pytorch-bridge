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

#include "generated/backend/bucketize.h"

namespace habana {

std::shared_ptr<void> FillBucketizeParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_SearchSorted::Params);
  params->right = stack.at(3).toBool();
  return params;
}

OutputMetaDataVector BucketizeMeta(const at::Stack& stack) {
  std::vector<int64_t> outshape;
  if (stack.at(0).isTensor()) {
    outshape = stack_tensor(stack, 0).sizes().vec();
  } else {
    outshape = {1};
  }
  bool out_int32 = stack.at(2).toBool();

  OutputMetaData meta;
  meta.shape = outshape;
  meta.dtype = out_int32 ? torch::kInt32 : torch::kLong;
  return {meta};
}

} // namespace habana

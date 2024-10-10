/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/equal.h"

namespace habana {

std::shared_ptr<void> FillEqualParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_EqualPt::Params);
  auto self_sizes = stack_tensor(stack, 0).sizes();
  auto other_sizes = stack_tensor(stack, 1).sizes();
  params->forceFalse = self_sizes.size() != other_sizes.size();
  return params;
}

OutputMetaDataVector EqualMeta(const at::Stack&) {
  OutputMetaData meta;
  meta.shape = {};
  meta.dtype = c10::ScalarType::Bool;
  return {meta};
}

} // namespace habana

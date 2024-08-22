/******************************************************************************
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

#include "generated/backend/eq.h"
#include "generated/backend/ge.h"
#include "generated/backend/gt.h"
#include "generated/backend/le.h"
#include "generated/backend/lt.h"
#include "generated/backend/ne.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {
OutputMetaDataVector CompareMeta(const at::Stack& stack) {
  OutputMetaData meta;
  const at::Tensor self = stack_tensor(stack, 0);
  meta.shape = stack[1].isScalar()
      ? self.sizes().vec()
      : at::infer_size(self.sizes(), stack_tensor(stack, 1).sizes());
  meta.dtype = at::kBool;
  return {meta};
}

SharedMetaDataVector CompareEqSharedMeta(const at::Stack& stack) {
  return CompareSharedMeta(stack, "equal_fwd");
}

SharedMetaDataVector CompareGeSharedMeta(const at::Stack& stack) {
  return CompareSharedMeta(stack, "greater_equal_fwd");
}

SharedMetaDataVector CompareGtSharedMeta(const at::Stack& stack) {
  return CompareSharedMeta(stack, "greater_fwd");
}

SharedMetaDataVector CompareLeSharedMeta(const at::Stack& stack) {
  return CompareSharedMeta(stack, "less_equal_fwd");
}

SharedMetaDataVector CompareLtSharedMeta(const at::Stack& stack) {
  return CompareSharedMeta(stack, "less_fwd");
}

} // namespace habana

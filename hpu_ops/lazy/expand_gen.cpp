/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "habana_kernels/lazy_kernels_declarations.h"
#include "hpu_ops/expand.h"

using habana_lazy::LazyOp;

namespace habana {
at::Tensor expand(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    bool implicit) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "HpuOp expand :",
      " self=",
      to_string(self),
      " size=",
      to_string(size),
      " implicit=",
      to_string(implicit));

  LazyOp<at::Tensor> hpu_op{"aten::expand", {self, size, implicit}};
  RUN_MAYBE_WITH_ACC_THREAD(expand, hpu_op);
}
} // namespace habana

/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************/
#include "habana_kernels/repeat.h"
#include "habana_eager/ops/eager_op.h"
namespace habana {
namespace eager {
at::Tensor repeat_hpu(const at::Tensor& self, at::SymIntArrayRef _repeats) {
  PT_EAGER_TRACE;
  auto repeats = C10_AS_INTARRAYREF_SLOW(_repeats);
  EagerOp<at::Tensor> k{
      "aten::repeat",
      {self, repeats},
      {RepeatOperator::compute_output_shape(self, repeats)},
      0};
  return k.call();
}
} // namespace eager
} // namespace habana

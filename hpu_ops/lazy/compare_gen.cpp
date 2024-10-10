/******************************************************************************
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

#include "generated/lazy/eq.h"
#include "generated/lazy/ge.h"
#include "generated/lazy/gt.h"
#include "generated/lazy/le.h"
#include "generated/lazy/lt.h"
#include "generated/lazy/ne.h"

namespace habana {
HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    CompareScalarToTensor,
    at::Tensor) {
  // convert scalar input to tensor to avoid cache misses in cases where scalar
  // value changes across iterations
  auto& x = get_inputs();
  auto self = x[0].toTensor();
  auto other = x[1].toScalar();
  auto dtype = at::result_type(self, other);
  x[1] = habana_lazy::get_tensor_for_scalar(
      other.toDouble(), self.options().dtype(dtype));
}
} // namespace habana

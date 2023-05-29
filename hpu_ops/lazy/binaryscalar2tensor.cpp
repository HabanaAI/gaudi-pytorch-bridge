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

#include "generated/lazy/div.h"

namespace habana {
HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    BinaryScalarToTensor_Int2Float,
    at::Tensor) {
  // convert scalar input to tensor to avoid cache misses in cases where scalar
  // value changes across iterations
  auto& x = get_inputs();
  auto self = x[0].toTensor();
  auto other = x[1].toScalar();
  auto dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      get_inputs(),
      c10::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteIntToFloat,
      false,
      c10::nullopt,
      false,
      false);
  x[1] = habana_lazy::get_tensor_for_scalar(
      other.toDouble(), self.options().dtype(dtype));
};
} // namespace habana
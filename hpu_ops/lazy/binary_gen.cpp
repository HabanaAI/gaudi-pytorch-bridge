/******************************************************************************
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

#include "generated/lazy/div.h"
#include "hpu_ops/common/scalar_dtype_range.h"

namespace habana {

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    BinaryScalarFE,
    at::Tensor&) {
  update_other_scalar_if_out_of_scalar_type_range(inputs, get_inputs());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    BinaryScalarFE,
    at::Tensor) {
  update_other_scalar_if_out_of_scalar_type_range(inputs, get_inputs());
}

} // namespace habana

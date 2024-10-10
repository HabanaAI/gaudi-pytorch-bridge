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

#include "generated/lazy/fill.h"
#include "hpu_ops/common/fill.h"

namespace habana {
HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(habana_lazy::LazyOp, FillFE, at::Tensor&) {
  CastBoolToInt(get_inputs());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(habana_lazy::LazyOp, FillFE, at::Tensor) {
  CastBoolToInt(get_inputs());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(habana_lazy::LazyOp, FillFE, at::Scalar&) {
  CastBoolToInt(get_inputs());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(habana_lazy::LazyOp, FillFE, at::Scalar) {
  CastBoolToInt(get_inputs());
}

} // namespace habana

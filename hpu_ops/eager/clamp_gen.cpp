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
 *******************************************************************************
 */

#include <c10_ver/core/SymIntArrayRef.h>
#include "generated/eager/clamp.h"
#include "generated/eager/clamp_max.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels_ver/wrap_kernels_declarations.h"
namespace habana {
HPU_OP_FRONTEND_CUSTOM_CTOR(eager::EagerOp, ClampFE, -1, at::Tensor) {}
HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, ClampFE, at::Tensor&) {}

HPU_OP_FRONTEND_CREATE_RESULT_ONLY(eager::EagerOp, ClampFE, at::Tensor) {
  auto& inputs = get_inputs();
  const auto& dtype = get_scalar_types()[0];
  const auto& t = inputs.at(0).toTensor();
  return hpu_wrap::empty(
      c10::fromIntArrayRefSlow(t.sizes()),
      dtype,
      t.options().layout_opt(),
      t.options().device_opt(),
      t.options().pinned_memory_opt(),
      t.suggest_memory_format());
}

} // namespace habana

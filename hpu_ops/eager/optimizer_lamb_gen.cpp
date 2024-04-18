/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include "hpu_ops/optimizer_lamb_gen.h"
#include "generated/eager/wrap_kernels_declarations.h"
#include "habana_helpers/dtype_helpers.h"

namespace habana {

HPU_OP_FRONTEND_CUSTOM_CTOR(
    eager::EagerOp,
    EagerOptimizerLambNorm,
    -1,
    at::Tensor) {}

HPU_OP_FRONTEND_CREATE_RESULT_ONLY(
    eager::EagerOp,
    EagerOptimizerLambNorm,
    at::Tensor) {
  const auto& inputs = get_inputs();
  const auto& t = inputs.at(0).toTensorList().get(0);

  return hpu_wrap::empty(
      {1},
      t.scalar_type(),
      t.options().layout_opt(),
      t.options().device_opt(),
      t.options().pinned_memory_opt(),
      t.suggest_memory_format());
}

} // namespace habana

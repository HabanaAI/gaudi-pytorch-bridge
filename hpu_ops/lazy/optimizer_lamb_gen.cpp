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

#include "hpu_ops/optimizer_lamb_gen.h"

namespace habana {

HPU_OP_FRONTEND_CUSTOM_CTOR(
    habana_lazy::LazyOp,
    LazyOptimizerLambNorm,
    -1,
    at::Tensor) {}

HPU_OP_FRONTEND_CREATE_RESULT_ONLY(
    habana_lazy::LazyOp,
    LazyOptimizerLambNorm,
    at::Tensor) {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(0).toTensorList().get(0);
  return habana_lazy::empty_hpu_lazy(
      {1}, t.options(), t.suggest_memory_format(), false);
}

} // namespace habana
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

namespace habana {

template <>
FillFE<at::Tensor&>::FillFE(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  // NOTE: Workaround due to https://github.com/pytorch/pytorch/issues/75465
  if (inputs[1].isBool()) {
    get_inputs()[1] = static_cast<int>(inputs[1].toBool());
  }
}

template <>
at::Tensor& FillFE<at::Tensor&>::get_result_overrideable() {
  return habana_lazy::LazyOp<at::Tensor&>::get_result_overrideable();
}
} // namespace habana

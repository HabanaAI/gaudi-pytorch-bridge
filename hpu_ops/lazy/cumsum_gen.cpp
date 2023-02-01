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

#include <utility>

#include "generated/lazy/cumprod.h"
#include "generated/lazy/cumsum.h"

namespace habana {

template <>
LazyCumsum<at::Tensor>::LazyCumsum(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
at::Tensor LazyCumsum<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(0).toTensor();
  const auto& options = inputs.at(2).isNone()
      ? isIntegralType(t.scalar_type(), true)
          ? t.options().dtype(c10::ScalarType::Long)
          : t.options()
      : t.options().dtype(inputs.at(2).toScalarType());
  return habana_lazy::empty_hpu_lazy(
      t.sizes(), options, t.suggest_memory_format(), false);
}
} // namespace habana

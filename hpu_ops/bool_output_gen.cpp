/*******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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
#include "generated/logical_and.h"
#include "generated/logical_not.h"
#include "generated/logical_or.h"
#include "generated/logical_xor.h"

namespace habana {
template <>
LazyBoolOutput<at::Tensor>::LazyBoolOutput(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
at::Tensor LazyBoolOutput<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(0).toTensor();
  auto shape =
      inputs.size() > 1 ? BinaryOutputShape(inputs)[0] : t.sizes().vec();
  return habana_lazy::empty_hpu_lazy(
      shape, t.options().dtype(at::kBool), t.suggest_memory_format(), false);
}

} // namespace habana

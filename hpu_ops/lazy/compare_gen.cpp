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
template <>
LazyCmp<at::Tensor>::LazyCmp(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {
  auto x = get_inputs();
  // convert scalar input to tensor to avoid cache misses in cases where scalar
  // value changes across iterations
  if (x[1].isScalar()) {
    auto self = x[0].toTensor();
    auto other = x[1].toScalar();
    auto dtype = at::result_type(self, other);
    auto other_tensor = habana_lazy::get_tensor_for_scalar(
        other.toDouble(), self.options().dtype(dtype));
    x[1] = c10::IValue(other_tensor);
    set_inputs(x);
  }
}

template <>
at::Tensor LazyCmp<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(0).toTensor();
  auto shape = BinaryOutputShape(inputs)[0];
  return habana_lazy::empty_hpu_lazy(
      shape, t.options().dtype(at::kBool), t.suggest_memory_format(), false);
}

} // namespace habana

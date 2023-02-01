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

#include "hpu_ops/common/bitwise_shift_gen.h"
#include "generated/lazy/bitwise_left_shift.h"
#include "generated/lazy/bitwise_right_shift.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

// To handle cases when scalar type !=tensor type
// type conversion similar to CPU implementation
// Jira raised for removing template specialization
// Jira link: https://jira.habana-labs.com/browse/SW-74866
template <>
ScalarTypeConversion<at::Tensor>::ScalarTypeConversion(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {
  auto x = get_inputs();
  if (x[index_of_self].isScalar()) {
    ScalarTypeConvert(x, index_of_self, index_of_other);
  } else {
    ScalarTypeConvert(x, index_of_other, index_of_self);
  }
  set_inputs(x);
}

template <>
at::Tensor ScalarTypeConversion<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  if (inputs.at(index_of_self).isScalar()) {
    const auto& t = inputs.at(index_of_other).toTensor();
    return habana_lazy::empty_hpu_lazy(
        t.sizes(), t.options(), t.suggest_memory_format(), false);
  } else {
    const auto& t = inputs.at(index_of_self).toTensor();
    return habana_lazy::empty_hpu_lazy(
        t.sizes(), t.options(), t.suggest_memory_format(), false);
  }
}
} // namespace habana

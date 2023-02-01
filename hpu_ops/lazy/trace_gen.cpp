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
#include "generated/lazy/trace.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

template <>
LazyTrace<at::Tensor>::LazyTrace(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  auto x = inputs.at(0).toTensor();
  // In CPU trace op promotes all int dtype input to Long.
  // Setting the HPU output to be of dtype = Long, as the CPU output for int
  // dtype input is Long.
  if (x.scalar_type() == c10::ScalarType::Int)
    set_scalar_type(c10::ScalarType::Long);
}

template <>
at::Tensor LazyTrace<at::Tensor>::get_result_overrideable() {
  HABANA_ASSERT(false, "Shouldn't be reachable");
  return {};
}

} // namespace habana

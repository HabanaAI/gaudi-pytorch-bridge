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

#include "generated/lazy/linalg_vector_norm.h"
#include "generated/lazy/norm.h"
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/lazy/reduction_template.h"

namespace habana {

template <>
LazyNormOp<at::Tensor>::LazyNormOp(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
at::Tensor LazyNormOp<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& self = inputs.at(0).toTensor();
  auto shape = get_out_shapes()[0];
  const at::ScalarType& dtype =
      inputs.at(4).isNone() ? self.scalar_type() : inputs.at(4).toScalarType();
  return habana_lazy::empty_hpu_lazy(
      shape, self.options().dtype(dtype), self.suggest_memory_format(), false);
}

} // namespace habana

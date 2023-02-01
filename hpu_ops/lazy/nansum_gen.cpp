/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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
#include "generated/lazy/nansum.h"
#include "habana_kernels/reduction_kernels.h"
#include "hpu_ops/lazy/reduction_template.h"

namespace habana {

template <>
LazyNansum<at::Tensor>::LazyNansum(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  if (!inputs.at(3).isNone())
    set_scalar_type(inputs[3].toScalarType());
}

template <>
at::Tensor LazyNansum<at::Tensor>::get_result_overrideable() {
  throw std::runtime_error("Shouldn't be invoked");
}

} // namespace habana

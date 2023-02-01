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

#include "generated/lazy/geometric.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {

template <>
LazyGeometric<at::Tensor&>::LazyGeometric(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  // Generators can't be represented in JIT graph
  // https://github.com/pytorch/pytorch/issues/64005
  LazyGeometric<at::Tensor&>::get_inputs().at(2) =
      get_seed_tensor_hpu(inputs.at(2).toOptional<at::Generator>());
}

template <>
at::Tensor& LazyGeometric<at::Tensor&>::get_result_overrideable() {
  return stack_tensor(LazyGeometric<at::Tensor&>::get_inputs(), 0);
}
} // namespace habana

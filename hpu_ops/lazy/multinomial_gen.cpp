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

#include "generated/lazy/multinomial.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_kernels/reduction_kernels.h"

namespace habana {

template <>
LazyRandomMulti<at::Tensor>::LazyRandomMulti(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {
  get_inputs().back() =
      get_seed_tensor_hpu(inputs.back().toOptional<at::Generator>());
}

template <>
at::Tensor LazyRandomMulti<at::Tensor>::get_result_overrideable() {
  auto t = get_inputs().at(0).toTensor();
  return habana_lazy::empty_hpu_lazy(
      get_out_shapes()[0],
      t.options().dtype(c10::ScalarType::Long),
      t.suggest_memory_format(),
      false);
}

template <>
LazyRandomMultiOut<at::Tensor&>::LazyRandomMultiOut(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn, -1) {
  // Seed is at the 3rd position
  get_inputs().at(3) =
      get_seed_tensor_hpu(inputs.at(3).toOptional<at::Generator>());
}

template <>
at::Tensor& LazyRandomMultiOut<at::Tensor&>::get_result_overrideable() {
  return stack_tensor(get_inputs(), 0);
}

} // namespace habana

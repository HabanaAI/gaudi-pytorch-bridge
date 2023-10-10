/******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#include <cstdint>
#include "generated/lazy/bitwise_and.h"
#include "habana_kernels/lazy_kernels.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {
static void bitwise_convert_scalar_to_tensor(
    std::vector<at::IValue>& inputs,
    const std::int64_t tensor_index,
    const std::int64_t scalar_index) {
  auto self = inputs[tensor_index].toTensor();
  auto other_tensor = habana_lazy::get_tensor_for_scalar(
      inputs[scalar_index].toScalar().toDouble(), self.scalar_type());
  inputs[scalar_index] = other_tensor;
}

template <typename T>
LazyBitwiseScalar<T>::LazyBitwiseScalar(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes_fn) {
  auto input = LazyBitwiseScalar<T>::get_inputs();
  bitwise_convert_scalar_to_tensor(
      input, 0 /*tensor_index*/, 1 /*tensor_index*/);
  LazyBitwiseScalar<T>::set_inputs(input);
}

template struct LazyBitwiseScalar<at::Tensor&>;
template struct LazyBitwiseScalar<at::Tensor>;

template <typename T>
T LazyBitwiseScalar<T>::get_result_overrideable() {
  HABANA_ASSERT(false, "Shouldn't be reachable");
  return habana_lazy::LazyOp<T>::get_result_overrideable();
}
} // namespace habana

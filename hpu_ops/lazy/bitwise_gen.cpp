/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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

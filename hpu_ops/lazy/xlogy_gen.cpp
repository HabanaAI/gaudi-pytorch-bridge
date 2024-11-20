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

#include "generated/lazy/special_xlog1py.h"
#include "generated/lazy/xlogy.h"

namespace habana {

constexpr size_t index_of_self = 0;
constexpr size_t index_of_other = 1;

static void XlogyScalarConversion(
    std::vector<at::IValue>& inputs,
    size_t scalar_index,
    size_t tensor_index) {
  auto tensor = inputs[tensor_index].toTensor();
  auto scalar = inputs[scalar_index].toScalar();
  auto dtype = at::result_type(tensor, scalar);
  auto self_tensor =
      habana_lazy::get_tensor_for_scalar(scalar.toDouble(), dtype);
  inputs[scalar_index] = c10::IValue(self_tensor);
}

template <typename T>
LazyXlogY<T>::LazyXlogY(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes_fn) {
  auto x = LazyXlogY<T>::get_inputs();
  // convert scalar input to tensor
  if (x[index_of_self].isScalar()) {
    XlogyScalarConversion(x, index_of_self, index_of_other);
  } else {
    XlogyScalarConversion(x, index_of_other, index_of_self);
  }
  LazyXlogY<T>::set_inputs(x);
}

template struct LazyXlogY<at::Tensor&>;
template struct LazyXlogY<at::Tensor>;

template <typename T>
T LazyXlogY<T>::get_result_overrideable() {
  HABANA_ASSERT(false, "Shouldn't be reachable");
  return habana_lazy::LazyOp<T>::get_result_overrideable();
}
} // namespace habana

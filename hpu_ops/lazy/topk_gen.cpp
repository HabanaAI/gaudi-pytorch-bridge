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
#include "generated/lazy/topk.h"

namespace habana {

static void convert_k_to_tensor(std::vector<at::IValue>& inputs) {
  auto self = inputs[0].toTensor();
  auto k = inputs[1].toScalar().to<int>();
  auto k_tensor = habana_lazy::empty_hpu_lazy(
      k, self.options(), self.suggest_memory_format(), false, SHAPE_TENSOR);
  inputs[1] = k_tensor;
}

template <>
LazyTopk<std::tuple<at::Tensor, at::Tensor>>::LazyTopk(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<std::tuple<at::Tensor, at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn) {
  auto input = get_inputs();
  convert_k_to_tensor(input);
  set_inputs(input);
}

template <>
std::tuple<at::Tensor, at::Tensor> LazyTopk<
    std::tuple<at::Tensor, at::Tensor>>::get_result_overrideable() {
  auto t = get_inputs().at(0).toTensor();
  auto out_shape = get_out_shapes()[0];
  at::Tensor values = habana_lazy::empty_hpu_lazy(
      out_shape, t.options(), t.suggest_memory_format(), false);
  at::Tensor indices = habana_lazy::empty_hpu_lazy(
      out_shape,
      t.options().dtype(c10::ScalarType::Long),
      t.suggest_memory_format(),
      false);
  return {values, indices};
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    TopKFE,
    std::tuple<at::Tensor&, at::Tensor&>) {
  convert_k_to_tensor(get_inputs());
}
} // namespace habana

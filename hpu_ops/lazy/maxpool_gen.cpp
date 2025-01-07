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
#include "generated/eager/max_pool3d_with_indices.h"
#include "generated/lazy/max_pool2d_with_indices.h"
#include "generated/lazy/max_pool2d_with_indices_backward.h"

namespace habana {

template <>
LazyMaxPool<std::tuple<at::Tensor, at::Tensor>>::LazyMaxPool(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<std::tuple<at::Tensor, at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn) {}

template <>
std::tuple<at::Tensor, at::Tensor> LazyMaxPool<
    std::tuple<at::Tensor, at::Tensor>>::get_result_overrideable() {
  auto inputs = get_inputs();
  auto t = inputs.at(0).toTensor();
  auto out_shape = get_out_shapes()[0];
  at::Tensor maxpool = habana_lazy::empty_hpu_lazy(
      out_shape, t.options(), t.suggest_memory_format(), false);
  // TODO Analyse memory performance for the dtype change
  // (https://jira.habana-labs.com/browse/SW-108396),
  at::Tensor indices = habana_lazy::empty_hpu_lazy(
      out_shape,
      t.options().dtype(c10::ScalarType::Long),
      t.suggest_memory_format(),
      false);
  return {maxpool, indices};
}

} // namespace habana

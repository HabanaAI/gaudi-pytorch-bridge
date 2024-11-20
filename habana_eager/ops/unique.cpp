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

#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/view.h"
#include "hpu_ops/unique.h"

namespace habana {
namespace eager {

::std::tuple<at::Tensor, at::Tensor> _unique_eager(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse) {
  int elements = self.numel();
  auto inputShape = self.sizes().vec();
  std::vector<int64_t> output_shape{elements};
  std::vector<int64_t> valid_count_shape{1};
  at::Tensor result{};
  at::Tensor inverse_tensor{};

  if (elements == 0) {
    result =
        at::empty(output_shape, self.options(), self.suggest_memory_format());
    if (return_inverse) {
      inverse_tensor = at::empty(
          inputShape,
          self.options().dtype(c10::ScalarType::Long),
          self.suggest_memory_format());
    }
    return {result, inverse_tensor};
  }

  if (return_inverse) {
    auto hpu_op =
        habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>>{
            "hpu::_unique_eager",
            {self, sorted, return_inverse},
            {output_shape, valid_count_shape, output_shape},
            0};

    hpu_op.SetOutputMetaFn(Unique2Meta);
    auto result_unique = hpu_op.call();
    auto feature_map = std::get<0>(result_unique);
    auto valid_count = std::get<1>(result_unique);
    auto end = valid_count.item<int64_t>();
    result = at::slice(feature_map, 0, 0, end, 1);

    if (!sorted) {
      result = at::flip(result, {0});
    }
    inverse_tensor = std::get<2>(result_unique);
    if (!sorted) {
      at::Tensor subtracter = at::add(valid_count, 1, -1);
      inverse_tensor = at::add(subtracter, inverse_tensor, -1);
    }
    inverse_tensor =
        view_hpu(inverse_tensor, fromIntArrayRefUnchecked(self.sizes()));
  } else {
    auto hpu_op = habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor>>{
        "hpu::_unique_eager",
        {self, sorted, return_inverse},
        {output_shape, valid_count_shape},
        0};

    hpu_op.SetOutputMetaFn(Unique2Meta);
    auto result_unique = hpu_op.call();
    auto feature_map = std::get<0>(result_unique);
    auto valid_count = std::get<1>(result_unique);
    auto end = valid_count.item<int64_t>();
    result = at::slice(feature_map, 0, 0, end, 1);
    if (!sorted) {
      result = at::flip(result, {0});
    }
  }
  return std::make_tuple(result, inverse_tensor);
}

TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def(
      "_unique_eager(Tensor self, bool sorted, bool return_inverse) -> (Tensor, Tensor, Tensor)");
}
} // namespace eager
} // namespace habana

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

#include "generated/lazy/upsample_linear1d.h"
#include "generated/lazy/upsample_linear1d_backward.h"
#include "generated/lazy/upsample_nearest1d.h"
#include "generated/lazy/upsample_nearest1d_backward.h"
#include "generated/lazy/upsample_nearest2d.h"
#include "generated/lazy/upsample_nearest2d_backward.h"
#include "generated/lazy/upsample_nearest3d.h"
#include "generated/lazy/upsample_nearest3d_backward.h"

namespace habana {
// FrontEnd
template <>
LazyUpsample<at::Tensor>::LazyUpsample(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  auto x = get_inputs();
  auto self = x[0].toTensor();
  auto meta = UpsampleNearest2DBwdMeta(x)[0];
  auto outshape_st = habana_lazy::empty_hpu_lazy(
      meta.shape,
      self.options(),
      self.suggest_memory_format(),
      false,
      SHAPE_TENSOR);

  x[2] = c10::IValue(outshape_st);
  set_inputs(x);
}

template <>
at::Tensor LazyUpsample<at::Tensor>::get_result_overrideable() {
  return LazyOp<at::Tensor>::get_result_overrideable();
}

template struct LazyUpsample<at::Tensor>;
} // namespace habana

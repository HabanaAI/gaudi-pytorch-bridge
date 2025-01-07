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

#include "hpu_ops/common/bitwise_shift_gen.h"
#include "generated/lazy/bitwise_left_shift.h"
#include "generated/lazy/bitwise_right_shift.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

// To handle cases when scalar type !=tensor type
// type conversion similar to CPU implementation
// Jira raised for removing template specialization
// Jira link: https://jira.habana-labs.com/browse/SW-74866
template <>
ScalarTypeConversion<at::Tensor>::ScalarTypeConversion(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {
  auto x = get_inputs();
  if (x[index_of_self].isScalar()) {
    ScalarTypeConvert(x, index_of_self, index_of_other);
  } else {
    ScalarTypeConvert(x, index_of_other, index_of_self);
  }
  set_inputs(x);
}

template <>
at::Tensor ScalarTypeConversion<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  if (inputs.at(index_of_self).isScalar()) {
    const auto& t = inputs.at(index_of_other).toTensor();
    return habana_lazy::empty_hpu_lazy(
        t.sizes(), t.options(), t.suggest_memory_format(), false);
  } else {
    const auto& t = inputs.at(index_of_self).toTensor();
    return habana_lazy::empty_hpu_lazy(
        t.sizes(), t.options(), t.suggest_memory_format(), false);
  }
}
} // namespace habana

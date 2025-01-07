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

#include "habana_helpers/dtype_helpers.h"
#include "lazy_kernels.h" // TODO rename to lazy_op.h

namespace habana_lazy {

template <>
at::Tensor LazyBinaryOp<at::Tensor>::get_result_overrideable() {
  const auto& inputs = LazyOp<at::Tensor>::get_inputs();
  const auto& self = inputs.at(0).toTensor();

  const auto& outshape = LazyOp<at::Tensor>::get_out_shapes().empty()
      ? self.sizes()
      : LazyOp<at::Tensor>::get_out_shapes().at(0);

  if (dst_dtype_ == at::ScalarType::Undefined) {
    c10::optional<const at::IValue*> output = is_outfn_
        ? c10::make_optional<const at::IValue*>(&inputs.back())
        : c10::nullopt;
    auto dtype_helper =
        habana_helpers::DTypeHelper::binary_op_with_type_promotion(
            inputs, output, safe_cast_check_);

    dst_dtype_ = dtype_helper.get_result_dtype();
  }

  return empty_hpu_lazy(
      outshape,
      self.options().device(c10::kHPU).dtype(dst_dtype_),
      self.suggest_memory_format(),
      false);
}

template <>
at::Tensor& LazyBinaryOp<at::Tensor&>::get_result_overrideable() {
  return LazyOp<at::Tensor&>::get_result_overrideable();
}

} // namespace habana_lazy

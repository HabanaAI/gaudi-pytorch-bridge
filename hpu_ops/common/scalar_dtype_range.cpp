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

#include "hpu_ops/common/scalar_dtype_range.h"
#include "habana_helpers/logging.h"

namespace habana {

template <typename T>
bool is_out_of_dtype_range(const float value) {
  const float abs_value = abs(value);
  return abs_value < float(std::numeric_limits<T>::min()) ||
      abs_value > float(std::numeric_limits<T>::max());
}

bool is_value_out_of_scalar_range(
    const float value,
    const c10::ScalarType scalar_type) {
  if (value == 0.0) {
    return false;
  }
  return ((scalar_type == torch::kBFloat16) &&
          is_out_of_dtype_range<c10::BFloat16>(value)) ||
      ((scalar_type == torch::kFloat16) &&
       is_out_of_dtype_range<c10::Half>(value));
}

void update_other_scalar_if_out_of_scalar_type_range(
    const std::vector<at::IValue>& inputs,
    std::vector<at::IValue>& hpu_inputs) {
  HABANA_ASSERT(inputs.size() >= 2, "There must be at least 2 inputs.");

  if (!inputs.at(0).isTensor() || !inputs.at(1).isTensor()) {
    return;
  }

  const auto& self = inputs.at(0).toTensor();
  const auto& other = inputs.at(1).toTensor();
  const c10::ScalarType self_type = self.scalar_type();

  if ((self_type != torch::kBFloat16 && self_type != torch::kFloat16) ||
      !other.unsafeGetTensorImpl()->is_wrapped_number() || !other.is_cpu()) {
    return;
  }

  const float value = other.item<float>();
  if (is_value_out_of_scalar_range(value, self_type)) {
    hpu_inputs.at(1) = value;
  }
}

} // namespace habana
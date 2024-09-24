/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
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

#pragma once

#include <torch/torch.h>

namespace habana {

template <typename T>
inline bool is_out_of_dtype_range(const float value) {
  return value < float(std::numeric_limits<T>::min()) ||
      value > float(std::numeric_limits<T>::max());
}

inline bool is_value_out_of_scalar_range(
    const float value,
    const c10::ScalarType scalar_type) {
  return ((scalar_type == torch::kBFloat16) &&
          is_out_of_dtype_range<c10::BFloat16>(value)) ||
      ((scalar_type == torch::kFloat16) &&
       is_out_of_dtype_range<c10::Half>(value));
}

void update_other_scalar_if_out_of_scalar_type_range(
    const std::vector<at::IValue>& inputs,
    std::vector<at::IValue>& hpu_inputs);

} // namespace habana
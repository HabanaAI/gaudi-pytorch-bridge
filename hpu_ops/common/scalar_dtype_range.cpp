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
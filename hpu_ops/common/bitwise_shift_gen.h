/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/hpu_op_helper.h"

namespace habana {

constexpr size_t index_of_self = 0;
constexpr size_t index_of_other = 1;
static inline void ScalarTypeConvert(
    std::vector<at::IValue>& inputs,
    size_t scalar_index,
    size_t tensor_index) {
  auto tensor = inputs[tensor_index].toTensor();
  auto scalar = inputs[scalar_index].toScalar();
  if (c10::isFloatingType(tensor.scalar_type())) {
    inputs[scalar_index] = scalar.to<float>();
  } else {
    inputs[scalar_index] = scalar.to<int>();
  }
}
} // namespace habana

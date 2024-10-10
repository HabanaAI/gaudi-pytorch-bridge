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

#include "generated/backend/div.h"
#include "habana_helpers/dtype_helpers.h"

// For use in div_rounding_mode
#define StrModeFloor "floor"
#define StrModeTruncate "trunc"

namespace habana {

static c10::ScalarType GetResultDtype(
    const std::vector<at::IValue>& inputs,
    bool int_to_float) {
  return habana_helpers::DTypeHelper::
      binary_op_with_optional_int_to_float_promotion(
             inputs, int_to_float, c10::nullopt, false)
          .get_result_dtype();
}

static c10::ScalarType GetCommonDtype(
    const std::vector<at::IValue>& inputs,
    bool int_to_float) {
  return habana_helpers::DTypeHelper::
      binary_op_with_optional_int_to_float_promotion(
             inputs, int_to_float, c10::nullopt, false)
          .get_common_dtype();
}

} // namespace habana

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

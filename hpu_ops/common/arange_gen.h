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
#include "hpu_ops/hpu_op_helper.h"
namespace habana {

inline bool can_convert(const c10::Scalar& value) {
  if (value.isFloatingPoint()) {
    auto float_value = value.toFloat();
    auto int_value = value.toInt();
    auto diff = float_value - int_value;
    return !(diff > 0);
  }
  return true;
}

} // namespace habana

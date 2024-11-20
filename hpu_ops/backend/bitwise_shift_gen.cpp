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
#include "generated/backend/bitwise_left_shift.h"
#include "generated/backend/bitwise_right_shift.h"

namespace habana {

static std::shared_ptr<void> FillBitwiseShiftParams(
    const at::Stack&,
    ShiftDir_t shift_dir,
    size_t& size) {
  PARAMS_STUB(ns_BitShiftKernel::Params);
  params->direction = shift_dir;
  return params;
}

void ValidateBitwiseShiftInputShapes(const at::Stack& stack) {
  if (stack[0].isTensor() && stack[1].isTensor()) {
    auto input_t_shape = stack[0].toTensor().sizes().vec();
    auto shift_t_shape = stack[1].toTensor().sizes().vec();
    std::reverse(input_t_shape.begin(), input_t_shape.end());
    std::reverse(shift_t_shape.begin(), shift_t_shape.end());

    auto min_rank = std::min(shift_t_shape.size(), input_t_shape.size());

    for (size_t i = 0; i < min_rank; i++) {
      if ((shift_t_shape[i] != 1) && (input_t_shape[i] != 1)) {
        TORCH_CHECK(
            shift_t_shape[i] == input_t_shape[i],
            "Input shape incompatible inputs at dim=",
            i,
            ", input1=",
            input_t_shape,
            ", input2=",
            shift_t_shape);
      }
    }
  } else {
    PT_BRIDGE_WARN("BitwiseShift Ops inputs are not tensors!");
  }
}

std::shared_ptr<void> FillLeftShiftParams(
    const at::Stack& stack,
    size_t& size) {
  ValidateBitwiseShiftInputShapes(stack);
  return FillBitwiseShiftParams(stack, ShiftDir_t::LEFT, size);
}

std::shared_ptr<void> FillRightShiftParams(
    const at::Stack& stack,
    size_t& size) {
  ValidateBitwiseShiftInputShapes(stack);
  return FillBitwiseShiftParams(stack, ShiftDir_t::RIGHT, size);
}

} // namespace habana

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

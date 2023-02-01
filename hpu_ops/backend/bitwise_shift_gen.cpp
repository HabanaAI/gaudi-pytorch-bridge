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
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

static std::shared_ptr<void> FillBitwiseShiftParams(
    const at::Stack&,
    ShiftDir_t shift_dir,
    size_t& size) {
  PARAMS_STUB(ns_BitShiftKernel::Params);
  params->direction = shift_dir;
  return params;
}

std::shared_ptr<void> FillLeftShiftParams(
    const at::Stack& stack,
    size_t& size) {
  return FillBitwiseShiftParams(stack, ShiftDir_t::LEFT, size);
}

std::shared_ptr<void> FillRightShiftParams(
    const at::Stack& stack,
    size_t& size) {
  return FillBitwiseShiftParams(stack, ShiftDir_t::RIGHT, size);
}

} // namespace habana

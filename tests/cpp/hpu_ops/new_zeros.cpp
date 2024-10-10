/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "util.h"
class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, new_zeros) {
  GenerateIntInputs(1, {{10, 3, 2}}, 0, 100);
  torch::IntArrayRef targetShape = {2, 4, 5};
  auto cpuTensor = GetCpuInput(0).to(torch::kLong);
  auto hpuTensor = GetHpuInput(0).to(torch::kLong);
  auto expected = at::native::new_zeros(cpuTensor, targetShape);
  auto result = at::native::new_zeros(hpuTensor, targetShape);
  Compare(expected, result);
}

TEST_F(HpuOpTest, new_zeros_with_dtype) {
  GenerateInputs(1, {{10, 3, 2}}, {torch::kFloat32});
  torch::IntArrayRef targetShape = {2, 4, 5};
  auto cpuTensor = GetCpuInput(0);
  auto hpuTensor = GetHpuInput(0);
  auto targetType = torch::kBFloat16;
  auto expected = at::native::new_zeros(cpuTensor, targetShape, targetType);
  auto result = at::native::new_zeros(hpuTensor, targetShape, targetType);
  Compare(expected, result);
}
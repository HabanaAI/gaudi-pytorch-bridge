/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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

TEST_F(HpuOpTest, _masked_scale1) {
  GenerateInputs(2, {{28}, {28}});
  float scale = 5.6;

  auto expected = at::_masked_scale(GetHpuInput(0), GetHpuInput(1), scale);
  auto result = _masked_scale(GetHpuInput(0), GetHpuInput(1), scale);
  Compare(expected, result);
}

TEST_F(HpuOpTest, _masked_scale2) {
  GenerateInputs(2, {{1, 1, 8}, {1, 1, 8}}, {torch::kBFloat16});
  float scale = 0.6;

  auto expected = at::_masked_scale(GetHpuInput(0), GetHpuInput(1), scale);
  auto result = _masked_scale(GetHpuInput(0), GetHpuInput(1), scale);
  Compare(expected, result);
}
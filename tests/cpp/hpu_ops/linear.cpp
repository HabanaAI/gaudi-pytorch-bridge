/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, linear_3d) {
  GenerateInputs(2, {{8, 4, 12, 7}, {5, 7}}, {torch::kBFloat16});

  auto expected = torch::linear(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::linear(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}
TEST_F(HpuOpTest, linear_4d) {
  GenerateInputs(2, {{2, 4, 5, 7, 9}, {3, 9}});

  auto expected = torch::linear(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::linear(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}
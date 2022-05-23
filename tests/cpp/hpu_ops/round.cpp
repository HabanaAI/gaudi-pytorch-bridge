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

TEST_F(HpuOpTest, RoundDecimalsTest_PosDec) {
  GenerateInputs(1);

  auto expected = torch::round(GetCpuInput(0), 1);
  auto result = torch::round(GetHpuInput(0), 1);

  Compare(expected, result);
}

TEST_F(HpuOpTest, RoundDecimalsTest_NegDec) {
  GenerateInputs(1);

  auto expected = torch::round(GetCpuInput(0), -1);
  auto result = torch::round(GetHpuInput(0), -1);

  Compare(expected, result);
}

TEST_F(HpuOpTest, RoundDecimalsTest_ZeroDec) {
  GenerateInputs(1);

  auto expected = torch::round(GetCpuInput(0), 0);
  auto result = torch::round(GetHpuInput(0), 0);

  Compare(expected, result);
}

TEST_F(HpuOpTest, RoundDecimalsTest_bf16) {
  GenerateInputs(1, torch::kBFloat16);

  auto expected = torch::round(GetCpuInput(0), 1);
  auto result = torch::round(GetHpuInput(0), 1);

  Compare(expected, result, 0.1, 0.1);
}

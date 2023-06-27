/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "../utils/device_type_util.h"
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, matmul_5dx1d) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  GenerateInputs(2, {{8, 4, 12, 7, 3}, {3}}, {torch::kBFloat16});

  auto expected = torch::matmul(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::matmul(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}
TEST_F(HpuOpTest, matmul_4dx1d) {
  GenerateInputs(2, {{2, 4, 5, 7}, {7}});

  auto expected = torch::matmul(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::matmul(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}
TEST_F(HpuOpTest, matmul_5dx5d) {
  GenerateInputs(2, {{3, 2, 4, 5, 6}, {3, 2, 4, 6, 10}});

  auto expected = torch::matmul(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::matmul(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}
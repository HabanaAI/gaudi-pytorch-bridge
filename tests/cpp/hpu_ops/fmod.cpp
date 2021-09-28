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

TEST_F(HpuOpTest, fmod) {
  // TODO: avoid 0 in 2nd input to avoid divide by zero exception in cpu run
  // GenerateInputs(2, {{2, 3}, {2, 3}}, {torch::kInt, torch::kFloat});
  GenerateInputs(2, {{2, 1}, {2, 3}}, {torch::kLong, torch::kFloat});
  // GenerateInputs(2, {{2, 1}, {2, 100}}, {torch::kDouble, torch::kLong});
  auto exp = torch::fmod(GetCpuInput(0), GetCpuInput(1));
  auto res = torch::fmod(GetHpuInput(0), GetHpuInput(1));

  Compare(exp, res);
}

TEST_F(HpuOpTest, fmod_scalar) {
  GenerateIntInputs(1, {{2, 3, 3}}, -10000, 10000);
  auto exp = torch::fmod(GetCpuInput(0), 25);
  auto res = torch::fmod(GetHpuInput(0), 25);

  Compare(exp, res);
}

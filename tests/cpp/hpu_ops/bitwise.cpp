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

TEST_F(HpuOpTest, bitwise_not) {
  GenerateIntInputs(1, {{2, 3, 3}}, -10000, 10000);
  auto exp = torch::bitwise_not(GetCpuInput(0));
  auto res = torch::bitwise_not(GetHpuInput(0));

  Compare(exp, res);
}

TEST_F(HpuOpTest, bitwise_not_) {
  GenerateInputs(1, {10}, {torch::kBool});
  auto exp = GetCpuInput(0).bitwise_not_();
  auto res = GetHpuInput(0).bitwise_not_();

  Compare(exp, res);
}

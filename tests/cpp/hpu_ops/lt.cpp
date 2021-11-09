/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <stdexcept>
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, lt_scalar_) {
  GenerateInputs(1, torch::kInt);
  float compVal = -0.1f;

  GetCpuInput(0).lt_(compVal);
  GetHpuInput(0).lt_(compVal);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, lt_tensor_) {
  GenerateInputs(2, torch::kInt32);

  GetCpuInput(0).lt_(GetCpuInput(0.));
  GetHpuInput(0).lt_(GetHpuInput(0.));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, lt_scalar) {
  GenerateInputs(1, torch::kInt);
  float compVal = 0;

  auto exp = GetCpuInput(0).lt(compVal);
  auto res = GetHpuInput(0).lt(compVal);

  Compare(exp, res);
}

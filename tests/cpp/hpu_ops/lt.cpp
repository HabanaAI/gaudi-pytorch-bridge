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

TEST_F(HpuOpTest, lt_scalar) {
  GenerateInputs(1, torch::kFloat);
  float compVal = 1.1f;

  GetCpuInput(0).lt_(compVal);
  GetHpuInput(0).lt_(compVal);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, lt_tensor) {
  GenerateInputs(2, torch::kInt32);

  GetCpuInput(0).lt_(GetCpuInput(1));
  GetHpuInput(0).lt_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

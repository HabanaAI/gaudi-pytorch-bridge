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

TEST_F(HpuOpTest, ge_scalar) {
  GenerateInputs(1, torch::kFloat);
  int other = 0;

  GetCpuInput(0).ge_(other);
  GetHpuInput(0).ge_(other);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, ge_tensor) {
  GenerateInputs(2, {{2, 3, 4}, {2, 3, 4}});

  GetCpuInput(0).ge_(GetCpuInput(1));
  GetHpuInput(0).ge_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

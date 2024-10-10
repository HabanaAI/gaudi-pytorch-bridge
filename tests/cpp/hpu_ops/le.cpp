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

TEST_F(HpuOpTest, le_scalar) {
  GenerateInputs(1, {{2, 3, 4}});
  float other = 1.1;

  GetCpuInput(0).le_(other);
  GetHpuInput(0).le_(other);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, le_tensor) {
  GenerateInputs(2, {{2, 3, 4}, {2, 3, 4}});

  GetCpuInput(0).le_(GetCpuInput(1));
  GetHpuInput(0).le_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

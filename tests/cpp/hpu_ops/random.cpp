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

TEST_F(HpuOpTest, random_) {
  GenerateInputs(1);

  SetSeed();
  auto result1 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  GenerateInputs(1);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  Compare(result1, result2);
}

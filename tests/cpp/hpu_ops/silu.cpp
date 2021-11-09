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

TEST_F(HpuOpTest, silu_) {
  GenerateInputs(1, torch::kBFloat16);

  auto expected = torch::silu_(GetCpuInput(0));
  auto result = torch::silu_(GetHpuInput(0));

  /*Bug ticket for high tolerance used:
  https://jira.habana-labs.com/browse/PO-510 */

  Compare(expected, result, 0.32, 1e-3);
}
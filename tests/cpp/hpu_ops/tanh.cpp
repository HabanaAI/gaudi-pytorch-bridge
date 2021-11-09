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

TEST_F(HpuOpTest, tanh_backward_out) {
  GenerateInputs(2);
  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");
  torch::tanh_backward_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::tanh_backward_outf(GetHpuInput(0), GetHpuInput(1), result);
  Compare(expected, result);
}

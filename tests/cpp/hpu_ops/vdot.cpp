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

TEST_F(HpuOpTest, vdot) {
  GenerateInputs(2, {{12}, {12}});
  auto expected = torch::vdot(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::vdot(GetHpuInput(0), GetHpuInput(1));
  Compare(expected, result);
}

TEST_F(HpuOpTest, vdot_out) {
  GenerateInputs(2, {{10}, {10}});

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::vdot_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::vdot_outf(GetHpuInput(0), GetHpuInput(1), result);
  Compare(expected, result);
}
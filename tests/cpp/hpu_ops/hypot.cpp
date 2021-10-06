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

TEST_F(HpuOpTest, hypot) {
  GenerateInputs(2);

  auto expected = torch::hypot(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::hypot(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}

TEST_F(HpuOpTest, hypot_out) {
  GenerateInputs(2);

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::hypot_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::hypot_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, hypot_) {
  GenerateInputs(2);

  GetCpuInput(0).hypot_(GetCpuInput(1));
  GetHpuInput(0).hypot_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

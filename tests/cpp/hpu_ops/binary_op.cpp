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

TEST_F(HpuOpTest, mvOut) {
  GenerateInputs(3, {{2, 3}, {3}, {2}});
  auto expected =
      torch::mv_outf(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2));
  auto result = torch::mv_outf(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2));

  Compare(expected, result);
}

TEST_F(HpuOpTest, sub_out) {
  GenerateInputs(2, {{3, 4, 1}, {3, 1, 1}}, {torch::kFloat32, torch::kInt32});
  float alpha = 0.1;

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::sub_outf(GetCpuInput(0), GetCpuInput(1), alpha, expected);
  torch::sub_outf(GetHpuInput(0), GetHpuInput(1), alpha, result);

  Compare(expected, result);
}
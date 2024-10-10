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

TEST_F(HpuOpTest, heaviside_f32) {
  GenerateInputs(2, {{2, 2}, {2, 2}}, {torch::kFloat, torch::kFloat});

  auto expected = torch::heaviside(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::heaviside(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}

TEST_F(HpuOpTest, heaviside_i32) {
  GenerateInputs(2, {{2, 2}, {2, 2}}, {torch::kInt, torch::kInt});

  auto expected = torch::heaviside(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::heaviside(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}

TEST_F(HpuOpTest, heaviside_out) {
  GenerateInputs(2, {{4, 3}, {1}}, {torch::kFloat, torch::kFloat});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::heaviside_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::heaviside_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, heaviside_) {
  GenerateInputs(2, {{2, 2}, {2, 2}}, {torch::kFloat, torch::kFloat});

  GetCpuInput(0).heaviside_(GetCpuInput(1));
  GetHpuInput(0).heaviside_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

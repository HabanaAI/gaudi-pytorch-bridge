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

TEST_F(HpuOpTest, frac) {
  GenerateInputs(1, torch::kFloat);

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::frac(GetCpuInput(0));
  auto result = torch::frac(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, frac_) {
  GenerateInputs(1, torch::kFloat);

  GetCpuInput(0).frac_();
  GetHpuInput(0).frac_();

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, frac_out) {
  GenerateInputs(1, torch::kFloat);
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::frac_outf(GetCpuInput(0), expected);
  torch::frac_outf(GetHpuInput(0), result);

  Compare(expected, result);
}

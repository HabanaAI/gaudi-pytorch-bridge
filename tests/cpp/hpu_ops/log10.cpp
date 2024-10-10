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

TEST_F(HpuOpTest, log10) {
  GenerateInputs(1, {{8, 24, 24, 3}});

  auto expected = torch::log10(GetCpuInput(0));
  auto result = torch::log10(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, log10_) {
  GenerateInputs(1, {{64, 64, 128}}, {torch::kBFloat16});

  GetCpuInput(0).log10_();
  GetHpuInput(0).log10_();

  Compare(GetCpuInput(0), GetHpuInput(0), 1e-02, 1e-02);
}

TEST_F(HpuOpTest, log10_out) {
  GenerateInputs(1, {{2, 64, 24, 12, 2}});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::log10_outf(GetCpuInput(0), expected);
  torch::log10_outf(GetHpuInput(0), result);

  Compare(expected, result);
}
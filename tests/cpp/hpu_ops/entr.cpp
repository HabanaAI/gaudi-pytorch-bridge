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

TEST_F(HpuOpTest, special_entr) {
  GenerateInputs(1);
  auto expected = torch::special_entr(GetCpuInput(0));
  auto result = torch::special_entr(GetHpuInput(0));
  Compare(expected, result);
}

TEST_F(HpuOpTest, special_entr_out) {
  GenerateInputs(1);
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::special_entr_outf(GetCpuInput(0), expected);
  torch::special_entr_outf(GetHpuInput(0), result);
  Compare(expected, result);
}
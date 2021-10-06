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

TEST_F(HpuOpTest, signbit_out) {
  GenerateInputs(1, torch::kFloat);
  torch::ScalarType dtype = torch::kBool;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::signbit_outf(GetCpuInput(0), expected);
  torch::signbit_outf(GetHpuInput(0), result);

  Compare(expected, result);
}

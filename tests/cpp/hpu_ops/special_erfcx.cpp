/******************************************************************************
 * Copyright (C) 2024 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, special_erfcx_out) {
  GenerateInputs(1);
  torch::ScalarType dtype = torch::kFloat32;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::special_erfcx_outf(GetCpuInput(0), expected);
  torch::special_erfcx_outf(GetHpuInput(0), result);

  Compare(expected, result, 0.03, 0.005);
}

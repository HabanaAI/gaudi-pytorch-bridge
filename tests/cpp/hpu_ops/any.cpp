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

TEST_F(HpuOpTest, any_out) {
  GenerateInputs(1, {{4, 8, 16, 32}}, {torch::kBool});
  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::any_out(expected, GetCpuInput(0));
  torch::any_out(result, GetHpuInput(0));

  Compare(expected, result);
}
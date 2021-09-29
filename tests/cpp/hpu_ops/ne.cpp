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

TEST_F(HpuOpTest, neScalar_out) {
  GenerateInputs(1, torch::kFloat);
  float compVal = 1.1f;
  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::ne_outf(GetCpuInput(0), compVal, expected);
  torch::ne_outf(GetHpuInput(0), compVal, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, neTensor_out) {
  GenerateInputs(2, torch::kInt32);

  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::ne_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::ne_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}

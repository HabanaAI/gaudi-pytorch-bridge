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

TEST_F(HpuOpTest, ne_scalar_out) {
  GenerateInputs(1, torch::kFloat);
  float compVal = -1.1f;
  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::ne_outf(GetCpuInput(0), compVal, expected);
  torch::ne_outf(GetHpuInput(0), compVal, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, ne_tensor_out) {
  GenerateInputs(2, {{1, 2, 1}, {2, 2, 1}}, {torch::kFloat, torch::kBFloat16});

  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::ne_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::ne_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, ne_scalar_inplace) {
  GenerateInputs(1, {{2, 64, 24, 12, 2}}, {torch::kInt});
  float other = 2.5;

  GetCpuInput(0).ne_(other);
  GetHpuInput(0).ne_(other);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, ne_tensor_inplace) {
  GenerateInputs(2);

  GetCpuInput(0).ne_(GetCpuInput(1));
  GetHpuInput(0).ne_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}
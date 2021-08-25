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

TEST_F(HpuOpTest, cumprod) {
  GenerateInputs(1);
  int dim = -1;
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::cumprod(GetCpuInput(0), dim, dtype);
  auto result = torch::cumprod(GetHpuInput(0), dim, dtype);

  Compare(expected, result);
}

TEST_F(HpuOpTest, cumprod_) {
  GenerateInputs(1, torch::kInt);
  int dim = 1;

  GetCpuInput(0).cumprod_(dim);
  GetHpuInput(0).cumprod_(dim);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, cumprod_out) {
  GenerateInputs(1, torch::kInt);
  int dim = 2;
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::cumprod_outf(GetCpuInput(0), dim, dtype, expected);
  torch::cumprod_outf(GetHpuInput(0), dim, dtype, result);

  Compare(expected, result);
}

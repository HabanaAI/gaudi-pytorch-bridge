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

TEST_F(HpuOpTest, prod_out) {
  GenerateInputs(1, {{3, 2, 2, 4}});
  torch::ScalarType dtype = torch::kFloat;
  int64_t dim = 3;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::prod_outf(GetCpuInput(0), dim, /*keepdim*/ true, dtype, expected);
  torch::prod_outf(GetHpuInput(0), dim, /*keepdim*/ true, dtype, result);
  Compare(expected, result);
}

// Check case for keepdim = false
TEST_F(HpuOpTest, prod_out_f) {
  GenerateInputs(1, {{4, 3, 3}});
  torch::ScalarType dtype = torch::kFloat;
  int64_t dim = -2;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::prod_outf(GetCpuInput(0), dim, /*keepdim*/ false, dtype, expected);
  torch::prod_outf(GetHpuInput(0), dim, /*keepdim*/ false, dtype, result);
  Compare(expected, result);
}
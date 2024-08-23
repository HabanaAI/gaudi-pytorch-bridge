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

TEST_F(HpuOpTest, mean_out_4d_float) {
  torch::ScalarType dtype = torch::kFloat;
  GenerateInputs(1, {{4, 2, 5, 2}});

  bool keepdim = false;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::mean_outf(GetCpuInput(0), {-1} /*dim*/, keepdim, {}, expected);
  torch::mean_outf(GetHpuInput(0), {-1} /*dim*/, keepdim, {}, result);

  Compare(expected, result);
}

/**
BFloat16 testcases fails with default tolerance
Issue Raised: https://jira.habana-labs.com/browse/SW-93267
**/

TEST_F(HpuOpTest, mean_out_3d_bfloat16) {
  torch::ScalarType dtype = torch::kBFloat16;
  GenerateInputs(1, dtype);

  bool keepdim = true;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::mean_outf(GetCpuInput(0), {2} /*dim*/, keepdim, {}, expected);
  torch::mean_outf(GetHpuInput(0), {2} /*dim*/, keepdim, {}, result);

  Compare(expected, result, 1e-2, 1e-2);
}

TEST_F(HpuOpTest, mean_out_5d_float) {
  torch::ScalarType dtype = torch::kBFloat16;
  GenerateInputs(1, {{6, 1, 3, 2, 5}}, torch::kFloat);

  bool keepdim = false;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::mean_outf(GetCpuInput(0), {1, 2, 3} /*dim*/, keepdim, dtype, expected);
  torch::mean_outf(GetHpuInput(0), {1, 2, 3} /*dim*/, keepdim, dtype, result);

  Compare(expected, result, 1e-2, 1e-2);
}

TEST_F(HpuOpTest, mean_dim_5d_float) {
  GenerateInputs(1, {{2, 5, 3, 4, 1}});
  bool keepdim = true;

  auto expected = torch::mean(GetCpuInput(0), {2} /*dim*/, keepdim, {});
  auto result = torch::mean(GetHpuInput(0), {2} /*dim*/, keepdim, {});

  Compare(expected, result);
}

TEST_F(HpuOpTest, mean_dim_bfloat) {
  GenerateInputs(1, torch::kBFloat16);
  torch::ScalarType dtype = torch::kFloat;
  bool keepdim = false;

  auto expected = torch::mean(GetCpuInput(0), {0} /*dim*/, keepdim, dtype);
  auto result = torch::mean(GetHpuInput(0), {0} /*dim*/, keepdim, dtype);

  Compare(expected, result, 1e-2, 1e-2);
}

TEST_F(HpuOpTest, mean_3d_float) {
  GenerateInputs(1, {{4, 6, 7}});
  torch::ScalarType dtype = torch::kBFloat16;

  auto expected = torch::mean(GetCpuInput(0), dtype);
  auto result = torch::mean(GetHpuInput(0), dtype);

  Compare(expected, result);
}

TEST_F(HpuOpTest, mean_bfloat) {
  GenerateInputs(1, torch::kBFloat16);

  auto expected = torch::mean(GetCpuInput(0));
  auto result = torch::mean(GetHpuInput(0));

  Compare(expected, result);
}
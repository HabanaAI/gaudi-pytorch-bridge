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

TEST_F(HpuOpTest, mm_float32) {
  GenerateInputs(2, {{2, 3}, {3, 3}});

  auto expected = torch::mm(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::mm(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}

/*
Note: Below testcase fails for BFloat16 with default tolerance
Issue raised : https://jira.habana-labs.com/browse/SW-71743
*/
TEST_F(HpuOpTest, mm_bfloat16) {
  torch::ScalarType dtype = torch::kBFloat16;

  GenerateInputs(2, {{30, 5}, {5, 25}}, dtype);

  auto expected = torch::mm(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::mm(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result, 1e-3, 1e-1);
}

TEST_F(HpuOpTest, mm_out_float32) {
  GenerateInputs(2, {{2, 3}, {3, 3}});

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::mm_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::mm_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}

/*
Note: Below testcase fails for BFloat16 with default tolerance
Issue raised : https://jira.habana-labs.com/browse/SW-71743
*/
TEST_F(HpuOpTest, mm_out_bfloat16) {
  torch::ScalarType dtype = torch::kBFloat16;

  GenerateInputs(2, {{30, 5}, {5, 25}}, dtype);

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::mm_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::mm_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result, 1e-03, 1e-01);
}
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

TEST_F(HpuOpTest, hardtanh_out) {
  GenerateInputs(1);
  int val = GenerateScalar();
  int max_val = (val > 0) ? val : -1 * val;
  int min_val = -1 * max_val;

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::hardtanh_outf(GetCpuInput(0), min_val, max_val, expected);
  torch::hardtanh_outf(GetHpuInput(0), min_val, max_val, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, hardtanh_bwd_out) {
  GenerateInputs(2);
  int val = GenerateScalar();
  int max_val = (val > 0) ? val : -1 * val;
  int min_val = -1 * max_val;

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::hardtanh_backward_outf(
      GetCpuInput(0), GetCpuInput(1), min_val, max_val, expected);
  torch::hardtanh_backward_outf(
      GetHpuInput(0), GetHpuInput(1), min_val, max_val, result);

  Compare(expected, result);
}
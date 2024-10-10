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

TEST_F(HpuOpTest, take) {
  GenerateInputs(1, {{4, 4}});
  auto float_cpu_input = GetCpuInput(0).clone();
  auto float_hpu_input = GetHpuInput(0).clone();

  GenerateIntInputs(1, {{4, 4}}, 0, 9);

  auto expected =
      torch::take(float_cpu_input, GetCpuInput(0).to(torch::kInt64));
  auto result = torch::take(float_hpu_input, GetHpuInput(0));
  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, take_out) {
  GenerateInputs(1, {{4, 4}}, {c10::ScalarType::BFloat16});
  auto float_cpu_input = GetCpuInput(0).clone();
  auto float_hpu_input = GetHpuInput(0).clone();

  GenerateIntInputs(1, {{4, 4}}, 0, 9);

  torch::ScalarType dtype = c10::ScalarType::BFloat16;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::take_outf(float_cpu_input, GetCpuInput(0).to(torch::kInt64), expected);
  torch::take_outf(float_hpu_input, GetHpuInput(0), result);
  Compare(expected, result, 0, 0);
}
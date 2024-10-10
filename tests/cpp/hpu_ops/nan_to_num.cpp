/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <limits>
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, nan_to_num_out) {
  GenerateInputs(1);

  auto cpu_input = GetCpuInput(0);
  auto inf = std::numeric_limits<float>::infinity();
  cpu_input = torch::where(
      cpu_input > 1,
      torch::tensor(std::numeric_limits<float>::quiet_NaN()),
      torch::where(
          cpu_input >= 0.1,
          torch::tensor(inf),
          torch::where(cpu_input <= -0.1, torch::tensor(-inf), cpu_input)));

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::nan_to_num_outf(
      cpu_input,
      /*nan*/ c10::nullopt,
      /*posinf*/ 6.9,
      /*neginf*/ 6.9,
      expected);

  auto hpu_input = cpu_input.to(torch::kHPU);
  torch::nan_to_num_outf(
      hpu_input,
      /*nan*/ c10::nullopt,
      /*posinf*/ 6.9,
      /*neginf*/ 6.9,
      result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, nan_to_num_usual) {
  GenerateInputs(1, {torch::kInt});

  auto cpu_input = GetCpuInput(0);
  cpu_input = torch::where(
      cpu_input > 1,
      torch::tensor(
          std::numeric_limits<int>::quiet_NaN(), torch::dtype(torch::kInt32)),
      cpu_input);

  auto expected = torch::nan_to_num(cpu_input);
  auto hpu_input = cpu_input.to(torch::kHPU);
  auto result = torch::nan_to_num(hpu_input);

  Compare(expected, result);
}

TEST_F(HpuOpTest, nan_to_num_) {
  GenerateInputs(1, {torch::kBFloat16});

  auto cpu_input = GetCpuInput(0);
  cpu_input = torch::where(
      cpu_input > 1,
      torch::tensor(
          std::numeric_limits<float>::quiet_NaN(),
          torch::dtype(torch::kBFloat16)),
      cpu_input);

  auto hpu_input = cpu_input.to(torch::kHPU);

  cpu_input.nan_to_num_(
      /*nan*/ 2.35,
      /*posinf*/ 3.22,
      /*neginf*/ 6.9);
  hpu_input.nan_to_num_(
      /*nan*/ 2.35,
      /*posinf*/ 3.22,
      /*neginf*/ 6.9);

  Compare(cpu_input, hpu_input);
}
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

TEST_F(HpuOpTest, frexp) {
  GenerateInputs(1, {{10}}, torch::kBFloat16);

  auto expected = torch::frexp(GetCpuInput(0));
  auto result = torch::frexp(GetHpuInput(0));

  Compare(std::get<0>(expected), std::get<0>(result));
  Compare(std::get<1>(expected), std::get<1>(result));
}

TEST_F(HpuOpTest, frexp_out) {
  GenerateInputs(1, {{10}});

  auto expected_mantissa = torch::empty(0, torch::kFloat32);
  auto expected_exponent = torch::empty(0, torch::kInt);

  auto result_mantissa = expected_mantissa.to(torch::kHPU);
  auto result_exponent = expected_exponent.to(torch::kHPU);

  torch::frexp_outf(GetCpuInput(0), expected_mantissa, expected_exponent);
  torch::frexp_outf(GetHpuInput(0), result_mantissa, result_exponent);

  Compare(expected_mantissa, result_mantissa);
  Compare(expected_exponent, result_exponent);
}
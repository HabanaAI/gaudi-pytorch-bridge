/******************************************************************************
 * Copyright (C) 2021-2024 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "../utils/device_type_util.h"
#include "util.h"

class FrexpHpuOpTest : public HpuOpTestUtil,
                       public testing::WithParamInterface<c10::ScalarType> {};

TEST_P(FrexpHpuOpTest, frexp) {
  const auto& dtype = GetParam();
  if (isGaudi() && dtype == torch::kFloat16) {
    GTEST_SKIP() << "Half dtype not supported on Gaudi1";
  }

  GenerateInputs(1, {{10}}, dtype);

  auto expected = torch::frexp(GetCpuInput(0));
  auto result = torch::frexp(GetHpuInput(0));

  Compare(std::get<0>(expected), std::get<0>(result));
  Compare(std::get<1>(expected), std::get<1>(result));
}

INSTANTIATE_TEST_SUITE_P(
    Frexp,
    FrexpHpuOpTest,
    testing::Values(torch::kFloat, torch::kFloat16, torch::kBFloat16));

TEST_F(FrexpHpuOpTest, frexp_out) {
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
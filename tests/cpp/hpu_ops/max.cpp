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

class MaxHpuOpTest
    : public HpuOpTestUtil,
      public testing::WithParamInterface<std::tuple<c10::ScalarType>> {};

TEST_P(MaxHpuOpTest, max) {
  const auto& testParams = GetParam();
  const auto dtype = std::get<0>(testParams);
  GenerateInputs(1, {dtype});
  auto exp = torch::max(GetCpuInput(0));
  auto res = torch::max(GetHpuInput(0));
  Compare(exp, res, 0, 0);
}

TEST_P(MaxHpuOpTest, max_other) {
  const auto& testParams = GetParam();
  const auto dtype = std::get<0>(testParams);
  GenerateInputs(2, {dtype});
  auto exp = torch::max(GetCpuInput(0), GetCpuInput(1));
  auto res = torch::max(GetHpuInput(0), GetHpuInput(1));
  Compare(exp, res, 0, 0);
}

TEST_P(MaxHpuOpTest, max_out) {
  const auto& testParams = GetParam();
  const auto dtype = std::get<0>(testParams);
  GenerateInputs(2, {dtype});
  auto exp = torch::empty(0, dtype);
  auto res = exp.to("hpu");
  torch::max_outf(GetCpuInput(0), GetCpuInput(1), exp);
  torch::max_outf(GetHpuInput(0), GetHpuInput(1), res);
  Compare(exp, res, 0, 0);
}

TEST_P(MaxHpuOpTest, max_out_with_broadcast) {
  const auto& testParams = GetParam();
  const auto dtype = std::get<0>(testParams);
  GenerateInputs(2, {{2, 1, 2}, {5, 2}}, {dtype});
  auto exp = torch::empty(0, dtype);
  auto res = exp.to("hpu");
  torch::max_outf(GetCpuInput(0), GetCpuInput(1), exp);
  torch::max_outf(GetHpuInput(0), GetHpuInput(1), res);
  Compare(exp, res, 0, 0);
}

INSTANTIATE_TEST_SUITE_P(
    max,
    MaxHpuOpTest,
    ::testing::Combine(
        ::testing::Values(torch::kFloat, torch::kBFloat16, torch::kInt)));

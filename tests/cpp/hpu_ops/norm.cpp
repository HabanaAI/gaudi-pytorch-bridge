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

class NormHpuOpTest
    : public HpuOpTestUtil,
      public testing::WithParamInterface<
          std::tuple<std::vector<int64_t>, float, c10::ScalarType>> {};

TEST_P(NormHpuOpTest, norm) {
  const auto& testParams = GetParam();
  auto values = std::get<0>(testParams);
  auto p = std::get<1>(testParams);
  const auto dtype = std::get<2>(testParams);

  torch::Tensor input = torch::rand(values);
  auto hinput = input.to(torch::kHPU);

  auto expected = torch::norm(input, p, dtype);
  auto result = torch::norm(hinput, p, dtype);
  Compare(expected, result);
}

INSTANTIATE_TEST_SUITE_P(
    norm,
    NormHpuOpTest,
    ::testing::Combine(
        ::testing::Values(
            std::vector<int64_t>({1, 10, 2, 3}),
            std::vector<int64_t>({10, 2, 2})),
        ::testing::Values<float>(0.5, 2),
        ::testing::Values(torch::kFloat)));

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, normd_out) {
  bool keepdim = false;
  torch::ScalarType dtype = torch::kBFloat16;
  GenerateInputs(1, {{2, 2, 4}}, {torch::kBFloat16});
  std::vector<int64_t> dims{0, -2};
  auto exp = torch::empty(0, dtype);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::norm_outf(GetCpuInput(0), 2, dims, keepdim, dtype, exp);
  torch::norm_outf(GetHpuInput(0), 2, dims, keepdim, dtype, res);
  Compare(exp, res);
}

TEST_F(HpuOpTest, normd_out_f32) {
  bool keepdim = true;
  torch::ScalarType dtype = torch::kFloat;
  GenerateInputs(1, {{8, 4, 4, 16}}, {torch::kFloat});
  std::vector<int64_t> dims{0, 2, 3};
  auto exp = torch::empty(0, dtype);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::norm_outf(GetCpuInput(0), 0.5, dims, keepdim, dtype, exp);
  torch::norm_outf(GetHpuInput(0), 0.5, dims, keepdim, dtype, res);
  Compare(exp, res);
}

TEST_F(HpuOpTest, norm_out_f32) {
  bool keepdim = true;
  torch::ScalarType dtype = torch::kFloat;
  GenerateInputs(1, {{8, 2, 4, 4, 16}}, {torch::kFloat});
  std::vector<int64_t> dims{1, 2, 3, 4};
  auto exp = torch::empty(0, dtype);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::norm_outf(GetCpuInput(0), 0.3, dims, keepdim, exp);
  torch::norm_outf(GetHpuInput(0), 0.3, dims, keepdim, res);
  Compare(exp, res);
}

TEST_F(HpuOpTest, norm) {
  GenerateInputs(1, {{8, 4, 4, 16}}, {torch::kFloat});

  auto expected = torch::norm(GetCpuInput(0), 0.5);
  auto result = torch::norm(GetHpuInput(0), 0.5);
  Compare(expected, result);
}
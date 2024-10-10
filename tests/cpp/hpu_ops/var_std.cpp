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

class VarStd : public HpuOpTestUtil {};

TEST_F(VarStd, var_out_keepdim4d) {
  GenerateInputs(1, {{5, 3, 3, 2, 2}});
  torch::ScalarType dtype = torch::kFloat;
  std::vector<int64_t> dim = {0, 3};
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::var_outf(GetCpuInput(0), dim, {}, true, expected);
  torch::var_outf(GetHpuInput(0), dim, {}, true, result);

  Compare(expected, result);
}

TEST_F(VarStd, var_out_keepdim) {
  GenerateInputs(1, {{4, 3, 3, 2}});
  torch::ScalarType dtype = torch::kFloat;
  std::vector<int64_t> dim = {0, 3};
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::var_outf(GetCpuInput(0), dim, {}, true, expected);
  torch::var_outf(GetHpuInput(0), dim, {}, true, result);

  Compare(expected, result);
}

TEST_F(VarStd, var_kfal5d) {
  GenerateInputs(1, {{5, 3, 3, 2, 2}});
  torch::ScalarType dtype = torch::kFloat;
  std::vector<int64_t> dim = {0, 3};

  auto expected = torch::var(GetCpuInput(0), dim, 1, false);
  auto result = torch::var(GetHpuInput(0), dim, 1, false);

  Compare(expected, result);
}

TEST_F(VarStd, var_kfal3d) {
  GenerateInputs(1, {{3, 3, 2}});
  torch::ScalarType dtype = torch::kFloat;
  std::vector<int64_t> dim = {0, 2};

  auto expected = torch::var(GetCpuInput(0), dim, 2, false);
  auto result = torch::var(GetHpuInput(0), dim, 2, false);

  Compare(expected, result);
}

TEST_F(VarStd, var_empty_dim) {
  GenerateInputs(1, {{5, 3, 3, 2, 2}});
  torch::ScalarType dtype = torch::kFloat;

  auto expected =
      torch::var(GetCpuInput(0), c10::optional<at::IntArrayRef>{}, 2);
  auto result = torch::var(GetHpuInput(0), c10::optional<at::IntArrayRef>{}, 2);

  Compare(expected, result);
}

/*
 * Below test will fail for BFloat16 for default tolerance
 * Issue raised: https://jira.habana-labs.com/browse/SW-94175
 */
TEST_F(VarStd, var_out_bf16) {
  GenerateInputs(1, {{4, 2, 6}}, {torch::kBFloat16});
  torch::ScalarType dtype = torch::kBFloat16;
  std::vector<int64_t> dim = {0, 1};
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::var_outf(GetCpuInput(0), dim, {}, true, expected);
  torch::var_outf(GetHpuInput(0), dim, {}, true, result);

  Compare(expected, result, 0.007, 0.006);
}

TEST_F(VarStd, var_mean_4d) {
  GenerateInputs(1, {{3, 4, 6, 8}});
  std::vector<int64_t> dim = {0, 3};

  auto exp = torch::var_mean(GetCpuInput(0), dim, 1, true);
  auto res = torch::var_mean(GetHpuInput(0), dim, 1, true);

  Compare(std::get<0>(exp), std::get<0>(res));
  Compare(std::get<1>(exp), std::get<1>(res));
}

TEST_F(VarStd, var_mean_4d_fal) {
  GenerateInputs(1, {{2, 4, 6, 8}});
  std::vector<int64_t> dim = {1, 3};

  auto exp = torch::var_mean(GetCpuInput(0), dim, {}, false);
  auto res = torch::var_mean(GetHpuInput(0), dim, {}, false);

  Compare(std::get<0>(exp), std::get<0>(res));
  Compare(std::get<1>(exp), std::get<1>(res));
}

/*
 * Below test will fail for BFloat16 for default tolerance
 * Issue raised: https://jira.habana-labs.com/browse/SW-94175
 */
TEST_F(VarStd, std_bf16) {
  GenerateInputs(1, {{3, 6, 5, 4}}, {torch::kBFloat16});

  std::vector<int64_t> dim = {1, 3};
  auto expected = torch::std(GetCpuInput(0), dim, 1, true);
  auto result = torch::std(GetHpuInput(0), dim, 1, true);

  Compare(expected, result, 6.1e-03, 1.4e-03);
}

TEST_F(VarStd, std_f32) {
  GenerateInputs(1, {{5, 3, 3, 2, 2}});
  torch::ScalarType dtype = torch::kFloat;
  std::vector<int64_t> dim = {0, 1, 2};
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::std_outf(GetCpuInput(0), dim, {}, true, expected);
  torch::std_outf(GetHpuInput(0), dim, {}, true, result);

  Compare(expected, result);
}

TEST_F(VarStd, std_mean) {
  GenerateInputs(1, {{3, 4, 6, 8}});
  std::vector<int64_t> dim = {0, 3};

  auto exp = torch::std_mean(GetCpuInput(0), dim, 1, true);
  auto res = torch::std_mean(GetHpuInput(0), dim, 1, true);

  Compare(std::get<0>(exp), std::get<0>(res));
  Compare(std::get<1>(exp), std::get<1>(res));
}

TEST_F(VarStd, std_meankfal) {
  GenerateInputs(1, {{2, 4, 6, 8}});
  std::vector<int64_t> dim = {1, 3};

  auto exp = torch::std_mean(GetCpuInput(0), dim, {}, false);
  auto res = torch::std_mean(GetHpuInput(0), dim, {}, false);

  Compare(std::get<0>(exp), std::get<0>(res));
  Compare(std::get<1>(exp), std::get<1>(res));
}

class VarStdWithParametrizedDim : public VarStd,
                                  public testing::WithParamInterface<int> {
 public:
  void RunDimensionsTest(
      const torch::ArrayRef<torch::IntArrayRef> dimSizes,
      const std::vector<int64_t>& selectedDims,
      bool keepdim = false) {
    GenerateInputs(dimSizes.size(), dimSizes);
    constexpr auto correction = 1;

    auto expected =
        torch::var(GetCpuInput(0), selectedDims, correction, keepdim);
    auto result = torch::var(GetHpuInput(0), selectedDims, correction, keepdim);

    Compare(expected, result);
  }
};

TEST_P(VarStdWithParametrizedDim, var_1dWithDifferentDimSizes) {
  const auto dimSize = GetParam();
  RunDimensionsTest({{dimSize}}, {0});
  RunDimensionsTest({{dimSize}}, {0}, true);
}

TEST_P(VarStdWithParametrizedDim, var_2dWithDifferentDimSizes) {
  const auto dimSize = GetParam();
  RunDimensionsTest({{dimSize, dimSize * 2}}, {0, -1});
  RunDimensionsTest({{dimSize, dimSize * 2}}, {0, -1}, true);
  RunDimensionsTest({{dimSize, dimSize * 3}}, {0, 1});
  RunDimensionsTest({{dimSize, dimSize * 3}}, {0, 1}, true);
}

TEST_P(VarStdWithParametrizedDim, var_3dWithDifferentDimSizes) {
  const auto dimSize = GetParam();
  RunDimensionsTest({{dimSize, dimSize, dimSize * 2}}, {-1, 0, 1});
  RunDimensionsTest({{dimSize, dimSize, dimSize * 2}}, {-1, 0, 1}, true);
}

INSTANTIATE_TEST_SUITE_P(
    VarStdWithParametrizedDim,
    VarStdWithParametrizedDim,
    ::testing::Values<int>(1, 2, 4, 8));
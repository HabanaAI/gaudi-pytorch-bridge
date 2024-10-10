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

class MedianHpuOpTest : public HpuOpTestUtil,
                        public testing::WithParamInterface<
                            std::tuple<c10::ScalarType, std::vector<int64_t>>> {
};

TEST_P(MedianHpuOpTest, median) {
  const auto& testParams = GetParam();
  auto dtype = std::get<0>(testParams);
  if (isGaudi() && dtype == torch::kFloat16) {
    GTEST_SKIP() << "Half dtype not supported on Gaudi1.";
  }
  auto size = std::get<1>(testParams);
  GenerateInputs(1, {{size}}, {dtype});
  auto expected = torch::median(GetCpuInput(0));
  auto result = torch::median(GetHpuInput(0));
  Compare(expected, result);
}
class MedianDimHpuOpTest
    : public HpuOpTestUtil,
      public testing::WithParamInterface<
          std::tuple<c10::ScalarType, std::vector<int64_t>, int64_t, bool>> {};
TEST_P(MedianDimHpuOpTest, median_dim) {
  /* Test sporadically failing on Gaudi3: SW-165423 */
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  const auto& testParams = GetParam();
  auto dtype = std::get<0>(testParams);
  if (isGaudi() && dtype == torch::kFloat16) {
    GTEST_SKIP() << "Half dtype not supported on Gaudi1.";
  }
  auto size = std::get<1>(testParams);
  auto axis = std::get<2>(testParams);
  auto keepdim = std::get<3>(testParams);
  GenerateInputs(1, {{size}}, {dtype});
  auto expected = torch::median(GetCpuInput(0), axis, keepdim);
  auto result = torch::median(GetHpuInput(0), axis, keepdim);
  Compare(std::get<0>(expected), std::get<0>(result));
}
class MedianDimValuesHpuOpTest : public HpuOpTestUtil,
                                 public testing::WithParamInterface<std::tuple<
                                     std::vector<int64_t>,
                                     std::vector<int64_t>,
                                     int64_t,
                                     bool>> {};
TEST_P(MedianDimValuesHpuOpTest, median_dim_values) {
  /* Test sporadically failing on Gaudi3: SW-167683 */
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  const auto& testParams = GetParam();
  auto size = std::get<0>(testParams);
  auto expected_size = std::get<1>(testParams);
  auto axis = std::get<2>(testParams);
  auto keepdim = std::get<3>(testParams);
  GenerateInputs(1, {{size}}, {torch::kFloat});
  auto expected_value = torch::empty({expected_size}, torch::kFloat);
  auto expected_index = torch::empty({expected_size}, torch::kLong);
  auto result_value = torch::empty(
      {expected_size}, torch::TensorOptions(torch::kFloat).device("hpu"));
  auto result_index = torch::empty(
      {expected_size}, torch::TensorOptions(torch::kLong).device("hpu"));
  torch::median_outf(
      GetCpuInput(0), axis, keepdim, expected_value, expected_index);
  torch::median_outf(GetHpuInput(0), axis, keepdim, result_value, result_index);
  Compare(expected_value, result_value);
}
/**
 * Test cases fail when the size of the tensor in the reduction axis >37
 * Issue Raised: https://jira.habana-labs.com/browse/SW-70095
 * Mis-match in the Median index value
 * Issue Raised: https://jira.habana-labs.com/browse/SW-68143
 */
INSTANTIATE_TEST_CASE_P(
    Median,
    MedianHpuOpTest,
    ::testing::Combine(
        ::testing::Values(
            torch::kFloat,
            torch::kFloat16,
            torch::kBFloat16,
            torch::kInt32),
        ::testing::Values(
            std::vector<int64_t>({24}),
            std::vector<int64_t>({17}),
            std::vector<int64_t>({6, 6}),
            std::vector<int64_t>({5, 7}),
            std::vector<int64_t>({2, 5, 3}),
            std::vector<int64_t>({1, 1, 1, 1}),
            std::vector<int64_t>({2, 3, 4, 1}),
            std::vector<int64_t>({2, 2, 2, 2, 2}))));
INSTANTIATE_TEST_CASE_P(
    MedianDim,
    MedianDimHpuOpTest,
    ::testing::Combine(
        ::testing::Values(
            torch::kFloat,
            torch::kFloat16,
            torch::kBFloat16,
            torch::kInt32),
        ::testing::Values(
            std::vector<int64_t>({8, 24, 24, 5}),
            std::vector<int64_t>({12, 8, 12, 8}),
            std::vector<int64_t>({64, 24, 4})),
        ::testing::Values(2, -2, 1),
        ::testing::Bool()));
INSTANTIATE_TEST_SUITE_P(
    MedianDimValues,
    MedianDimValuesHpuOpTest,
    testing::Values(
        std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t, bool>(
            {12, 4, 8, 16},
            {12, 4, 8, 1},
            3,
            true),
        std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t, bool>(
            {24, 8, 16},
            {8, 16},
            0,
            false),
        std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t, bool>(
            {16, 8, 16},
            {16, 8, 1},
            -1,
            true)));
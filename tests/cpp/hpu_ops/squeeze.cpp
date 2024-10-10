/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "util.h"

class SqueezeDimsOpTest : public HpuOpTestUtil,
                          public testing::WithParamInterface<std::tuple<
                              std::vector<int64_t>, // input shape
                              std::vector<int64_t>, // dims
                              c10::ScalarType, // dtype
                              bool>> {}; // modify view

TEST_P(SqueezeDimsOpTest, squeeze_dims) {
  const auto& testParams = GetParam();
  auto shape = std::get<0>(testParams);
  auto dims = std::get<1>(testParams);
  auto dtype = std::get<2>(testParams);
  auto modify_view = std::get<3>(testParams);
  GenerateInputs(1, {dtype}, {shape});

  auto expected = torch::squeeze(GetCpuInput(0), dims);
  auto result = torch::squeeze(GetHpuInput(0), dims);
  if (modify_view) {
    expected.add_(1);
    result.add_(1);
  }
  Compare(expected, result);
}

INSTANTIATE_TEST_SUITE_P(
    sanity,
    SqueezeDimsOpTest,
    ::testing::Combine(
        ::testing::Values(
            std::vector<int64_t>({3, 1, 7, 4, 1}),
            std::vector<int64_t>({1, 5, 1, 1, 8})),
        ::testing::Values(
            std::vector<int64_t>({0, 3}),
            std::vector<int64_t>({-1, 2}),
            std::vector<int64_t>({-2, 1, 0}),
            std::vector<int64_t>({})),
        ::testing::Values<c10::ScalarType>(
            torch::kFloat,
            torch::kBFloat16,
            torch::kInt),
        ::testing::Values<bool>(true, false)));

class SqueezeDimsOpTestDim0 : public HpuOpTestUtil,
                              public testing::WithParamInterface<std::tuple<
                                  c10::ScalarType, // dtype
                                  bool>> {}; // modify view

TEST_P(SqueezeDimsOpTestDim0, squeeze_dims) {
  const auto& testParams = GetParam();
  std::vector<int64_t> shape{1};
  std::vector<int64_t> dims{0};
  auto dtype = std::get<0>(testParams);
  auto modify_view = std::get<1>(testParams);
  GenerateInputs(1, {dtype}, {shape});

  auto expected = torch::squeeze(GetCpuInput(0), dims);
  auto result = torch::squeeze(GetHpuInput(0), dims);
  if (modify_view) {
    expected.add_(1);
    result.add_(1);
  }
  Compare(expected, result);
}

INSTANTIATE_TEST_SUITE_P(
    sanity,
    SqueezeDimsOpTestDim0,
    ::testing::Combine(
        ::testing::Values<c10::ScalarType>(
            torch::kFloat,
            torch::kBFloat16,
            torch::kInt),
        ::testing::Values<bool>(true, false)));

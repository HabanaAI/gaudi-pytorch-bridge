/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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

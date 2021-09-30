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

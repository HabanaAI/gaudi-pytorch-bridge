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
#include "../utils/dtype_supported_on_device.h"
#include "backend/habana_device/HPUGuardImpl.h"
#include "util.h"

class OneHotHpuOpTestFixture
    : public ::testing::TestWithParam<
          std::tuple<c10::ScalarType, std::vector<int64_t>>>,
      public HpuOpTestUtilBase

{};

TEST_P(OneHotHpuOpTestFixture, one_hot) {
  torch::ScalarType dtype = std::get<0>(GetParam());
  if (!IsDtypeSupportedOnCurrentDevice(dtype)) {
    GTEST_SKIP();
  }
  c10::ArrayRef sizes(std::get<1>(GetParam()));
  GenerateInputs(1, {sizes}, dtype);

  auto num_of_classes = std::max(
      GetCpuInput(0).max().item().toLong() + 1,
      abs(GetCpuInput(0).min().item().toLong()) + 1);
  auto expected_cpu =
      torch::one_hot(abs(GetCpuInput(0)), num_of_classes).to(dtype);
  auto expected_hpu =
      torch::one_hot(abs(GetHpuInput(0)), num_of_classes).to(dtype);
  Compare(expected_cpu, expected_hpu);
}

INSTANTIATE_TEST_CASE_P(
    OneHotHpuOpTest,
    OneHotHpuOpTestFixture,
    ::testing::Combine(
        ::testing::Values<c10::ScalarType>(torch::kLong),
        ::testing::Values<std::vector<int64_t>>(
            std::vector<int64_t>{4, 5},
            std::vector<int64_t>{2, 2, 2, 2})));
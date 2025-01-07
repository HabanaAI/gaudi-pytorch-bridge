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
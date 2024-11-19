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

#include <gtest/gtest.h>
#include "hpu_ops/common/scalar_dtype_range.h"

class IsValueOutOfScalarRangeTest
    : public ::testing::TestWithParam<std::tuple<at::ScalarType, float, bool>> {
 public:
  struct GetName {
    template <class ParamType>
    std::string operator()(
        const ::testing::TestParamInfo<ParamType>& info) const {
      ::std::stringstream ss;

      const auto [dtype, value, in_range] = info.param;
      ss << "dtype_" << dtype << "_value_" << value;

      std::string testName = ss.str();
      std::replace(testName.begin(), testName.end(), '-', '_');
      std::replace(testName.begin(), testName.end(), '+', '_');
      std::replace(testName.begin(), testName.end(), '.', 'p');

      return testName;
    }
  };
};

TEST_P(IsValueOutOfScalarRangeTest, is_value_out_of_range_test) {
  auto [dtype, value, in_range] = GetParam();

  ASSERT_EQ(habana::is_value_out_of_scalar_range(value, dtype), in_range);
}

INSTANTIATE_TEST_CASE_P(
    full,
    IsValueOutOfScalarRangeTest,
    ::testing::Values(
        std::make_tuple(torch::kBFloat16, 0, false),
        std::make_tuple(torch::kBFloat16, 65, false),
        std::make_tuple(torch::kBFloat16, -65, false),
        std::make_tuple(torch::kBFloat16, 3.4e+38, true),
        std::make_tuple(torch::kFloat16, 1.e-38, true),
        std::make_tuple(torch::kFloat16, 0, false),
        std::make_tuple(torch::kFloat16, 65, false),
        std::make_tuple(torch::kFloat16, -65, false),
        std::make_tuple(torch::kFloat16, 1e10, true),
        std::make_tuple(torch::kFloat16, -1e-10, true)),
    IsValueOutOfScalarRangeTest::GetName());
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

class HpuOpTest : public HpuOpTestUtil {};

// threshold and value are zero (hardcoded)
// check relu test case
TEST_F(HpuOpTest, threshold) {
  GenerateInputs(1, torch::kFloat);
  float threshold = 0;
  float value = 0;
  auto expected = torch::threshold(GetCpuInput(0), threshold, value);
  auto result = torch::threshold(GetHpuInput(0), threshold, value);
  Compare(expected, result);
}

TEST_F(HpuOpTest, threshold_) {
  GenerateInputs(1, torch::kBFloat16);
  float threshold = GenerateScalar<float>(0, 1);
  float value = GenerateScalar<float>();
  torch::threshold_(GetCpuInput(0), threshold, value);
  torch::threshold_(GetHpuInput(0), threshold, value);
  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, threshold_out) {
  GenerateInputs(1, torch::kBFloat16);
  float threshold = GenerateScalar<float>(0, 1);
  float value = GenerateScalar<float>();
  torch::ScalarType dtype = torch::kBFloat16;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::threshold_outf(GetCpuInput(0), threshold, value, expected);
  torch::threshold_outf(GetHpuInput(0), threshold, value, result);
  Compare(expected, result);
}

class ThresholdBwdOpTest : public HpuOpTestUtil,
                           public testing::WithParamInterface<std::tuple<
                               c10::ScalarType, // dtype
                               float>> {};

TEST_P(ThresholdBwdOpTest, threshold_backward_out) {
  GenerateInputs(2);
  const auto& testParams = GetParam();
  auto dtype = std::get<0>(testParams);
  auto threshold = std::get<1>(testParams);
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::threshold_backward_outf(
      GetCpuInput(0), GetCpuInput(1), threshold, expected);
  torch::threshold_backward_outf(
      GetHpuInput(0), GetHpuInput(1), threshold, result);
  Compare(expected, result);
}

INSTANTIATE_TEST_SUITE_P(
    Threshold,
    ThresholdBwdOpTest,
    ::testing::Combine(
        ::testing::Values<c10::ScalarType>(torch::kFloat, torch::kBFloat16),
        ::testing::Values<float>(-100, -0.5, 0, 0.5, 100)));
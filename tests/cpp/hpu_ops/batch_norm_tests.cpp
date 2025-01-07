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

#include "../utils/device_type_util.h"
#include "util.h"

class BatchNormHpuOpTest : public HpuOpTestUtil {};

class NativeBatchNormLegitNoStatsHpuOpTest
    : public BatchNormHpuOpTest,
      public ::testing::WithParamInterface<std::tuple<
          std::vector<long int>,
          torch::ScalarType,
          bool,
          float,
          float>> {
 public:
  void generateAndCompare(
      const torch::IntArrayRef sizes,
      const torch::ScalarType dtype,
      const bool training,
      const float momentum,
      const float epsilon) {
    GenerateInputs(3, {sizes, {sizes[1]}, {sizes[1]}}, {dtype});

    auto expected = torch::_native_batch_norm_legit(
        GetCpuInput(0),
        GetCpuInput(1),
        GetCpuInput(2),
        training,
        momentum,
        epsilon);
    auto result = torch::_native_batch_norm_legit(
        GetHpuInput(0),
        GetHpuInput(1),
        GetHpuInput(2),
        training,
        momentum,
        epsilon);

    Compare(expected, result);
  }
};

TEST_P(NativeBatchNormLegitNoStatsHpuOpTest, nativeBatchNormLegitNoStatsTest) {
  const auto [inputTensorShape, inputTensorType, training, momentum, epsilon] =
      GetParam();

  if (isGaudi2()) {
    GTEST_SKIP()
        << "Temporary test skipped on Gaudi2 due to https://jira.habana-labs.com/browse/SW-184291.";
  }

  if (not training)
    GTEST_SKIP(); // TODO [Jira: SW-150573] seg fault visible for CPU pytorch
                  // result

  generateAndCompare(
      inputTensorShape, inputTensorType, training, momentum, epsilon);
}

INSTANTIATE_TEST_CASE_P(
    BatchNormTests,
    NativeBatchNormLegitNoStatsHpuOpTest,
    ::testing::Values(
        std::make_tuple(
            std::vector<long int>{2, 3, 4, 5},
            at::kFloat,
            true,
            0.999f,
            1e-5f),
        std::make_tuple(
            std::vector<long int>{5, 4, 3, 2},
            at::kFloat,
            false,
            0.999f,
            1e-5f)));

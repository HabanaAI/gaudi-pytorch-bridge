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

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, matmul_5dx1d) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  GenerateInputs(2, {{8, 4, 12, 7, 3}, {3}}, {torch::kBFloat16});

  auto expected = torch::matmul(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::matmul(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}
TEST_F(HpuOpTest, matmul_4dx1d) {
  GenerateInputs(2, {{2, 4, 5, 7}, {7}});

  auto expected = torch::matmul(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::matmul(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}
TEST_F(HpuOpTest, matmul_5dx5d) {
  GenerateInputs(2, {{3, 2, 4, 5, 6}, {3, 2, 4, 6, 10}});

  auto expected = torch::matmul(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::matmul(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}
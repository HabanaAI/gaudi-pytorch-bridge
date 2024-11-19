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

TEST_F(HpuOpTest, cat) {
  GenerateInputs(3, {at::kFloat, at::kInt, at::kBFloat16});
  auto expected = torch::cat({GetCpuInput(0), GetCpuInput(1), GetCpuInput(2)});
  auto result = torch::cat({GetHpuInput(0), GetHpuInput(1), GetHpuInput(2)});
  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, cat_empty) {
  GenerateInputs(2, {{0}, {8, 3, 24, 24}});
  auto expected = torch::cat({GetCpuInput(0), GetCpuInput(1)}, 1);
  auto result = torch::cat({GetHpuInput(0), GetHpuInput(1)}, 1);
  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, DISABLED_cat_out) {
  // GenerateInputs(3, {at::kFloat, at::kInt, at::kBFloat16});
  GenerateInputs(3, {at::kInt, at::kBFloat16, at::kBFloat16});
  auto expected = torch::empty(0, at::kBFloat16);
  auto result = expected.to(at::kHPU);
  torch::cat_outf(
      {GetCpuInput(0), GetCpuInput(1), GetCpuInput(2)}, 0, expected);
  torch::cat_outf({GetHpuInput(0), GetHpuInput(1), GetHpuInput(2)}, 0, result);

  Compare(expected, result, 0, 0);
}

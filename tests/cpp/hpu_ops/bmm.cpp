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

TEST_F(HpuOpTest, bmm_f32) {
  GenerateInputs(2, {{4, 2, 3}, {4, 3, 5}});
  auto expected = torch::bmm(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::bmm(GetHpuInput(0), GetHpuInput(1));
  Compare(expected, result);
}

TEST_F(HpuOpTest, bmm_out_f32) {
  GenerateInputs(2, {{4, 2, 3}, {4, 3, 5}});
  auto expected = torch::empty({4, 2, 5}, torch::kFloat32);
  auto result = expected.to(torch::kHPU);
  torch::bmm_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::bmm_outf(GetHpuInput(0), GetHpuInput(1), result);
  Compare(expected, result);
}

/**
 * Default tolerance will fail for BFloat16
 * Issue Raised: https://jira.habana-labs.com/browse/SW-65157
 */
TEST_F(HpuOpTest, bmm_bf16) {
  GenerateInputs(2, {{4, 5, 6}, {4, 6, 10}}, torch::kBFloat16);
  auto expected = torch::bmm(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::bmm(GetHpuInput(0), GetHpuInput(1));
  Compare(expected, result, 2e-2, 2e-2);
}

TEST_F(HpuOpTest, bmm_out_bf16) {
  GenerateInputs(2, {{4, 5, 6}, {4, 6, 10}}, torch::kBFloat16);
  auto expected = torch::empty({4, 5, 10}, torch::kBFloat16);
  auto result = expected.to(torch::kHPU);
  torch::bmm_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::bmm_outf(GetHpuInput(0), GetHpuInput(1), result);
  Compare(expected, result, 2e-2, 2e-2);
}

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

TEST_F(HpuOpTest, log1p_usual) {
  GenerateInputs(1, {{8, 3, 24, 24}}, {torch::kFloat});

  auto expected = torch::log1p(GetCpuInput(0));
  auto result = torch::log1p(GetHpuInput(0));

  Compare(expected, result);
}

/**
 * Testcase Failed for Log1P BFloat16 Datatype
 * Issue raised: https://jira.habana-labs.com/browse/SW-63960

TEST_F(HpuOpTest, log1p_inplace) {
  GenerateInputs(1, {{3, 24, 24}}, {torch::kBFloat16});

  GetCpuInput(0).log1p_();
  GetHpuInput(0).log1p_();

  Compare(GetCpuInput(0), GetHpuInput(0));
}
 */

TEST_F(HpuOpTest, log1p_out) {
  GenerateInputs(2, {{64, 64}, {64, 64}}, {torch::kFloat, torch::kFloat});

  torch::log1p_outf(GetCpuInput(0), GetCpuInput(1));
  torch::log1p_outf(GetHpuInput(0), GetHpuInput(1));

  Compare(GetCpuInput(1), GetHpuInput(1));
}

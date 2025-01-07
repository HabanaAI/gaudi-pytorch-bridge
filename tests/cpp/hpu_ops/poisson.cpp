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

TEST_F(HpuOpTest, poisson_float) {
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);

  GenerateInputs(1, {{11}});
  auto result1 = torch::poisson(GetHpuInput(0), gen1);
  GenerateInputs(1, {{11}});
  auto result2 = torch::poisson(GetHpuInput(0), gen2);

  EXPECT_TRUE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, poisson_bfloat) {
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/7548627345108);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/7548627345108);

  GenerateInputs(1, {{6, 2}}, torch::kBFloat16);
  auto result1 = torch::poisson(GetHpuInput(0), gen1);
  GenerateInputs(1, {{6, 2}}, torch::kBFloat16);
  auto result2 = torch::poisson(GetHpuInput(0), gen2);

  EXPECT_TRUE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, poisson_diff_seed) {
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/34612421310721);

  GenerateInputs(1);
  auto result1 = torch::poisson(GetHpuInput(0), gen1);
  GenerateInputs(1);
  auto result2 = torch::poisson(GetHpuInput(0), gen2);

  EXPECT_FALSE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, poisson_input) {
  GenerateInputs(1);
  auto result1 = torch::poisson(GetHpuInput(0));
  GenerateInputs(1);
  auto result2 = torch::poisson(GetHpuInput(0));

  EXPECT_TRUE(torch::equal(result1, result2));
}
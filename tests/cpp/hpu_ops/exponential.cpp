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

TEST_F(HpuOpTest, exponential_inplace_f32_1) {
  double lambd = GenerateScalar<double>(1.0, 5.0);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);

  GenerateInputs(1, {{1024}});
  auto result1 = GetHpuInput(0).exponential_(lambd, gen1);

  GenerateInputs(1, {{1024}});
  auto result2 = GetHpuInput(0).exponential_(lambd, gen2);

  EXPECT_TRUE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_inplace_f32_diff_seed) {
  double lambd = GenerateScalar<double>(1.0, 5.0);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/41216728023107);

  GenerateInputs(1, {{1024}});
  auto result1 = GetHpuInput(0).exponential_(lambd, gen1);

  GenerateInputs(1, {{1024}});
  auto result2 = GetHpuInput(0).exponential_(lambd, gen2);

  EXPECT_FALSE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_inplace_f32_2) {
  double lambd = GenerateScalar<double>(5.0, 15.0);

  GenerateInputs(1, {{256, 256}});
  auto result1 = GetHpuInput(0).exponential_(lambd);

  GenerateInputs(1, {{256, 256}});
  auto result2 = GetHpuInput(0).exponential_(lambd);

  EXPECT_TRUE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_inplace_bf16_3) {
  double lambd = GenerateScalar<double>(3.0, 10.0);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/41216728023107);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/41216728023107);

  GenerateInputs(1, {{24, 32, 32}}, {torch::kBFloat16});
  auto result1 = GetHpuInput(0).exponential_(lambd, gen1);

  GenerateInputs(1, {{24, 32, 32}}, {torch::kBFloat16});
  auto result2 = GetHpuInput(0).exponential_(lambd, gen2);

  EXPECT_TRUE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_inplace_f32_4) {
  double lambd = GenerateScalar<double>(30.0, 80.0);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/16741280223107);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/16741280223107);

  GenerateInputs(1, {{8, 3, 24, 24}});
  auto result1 = GetHpuInput(0).exponential_(lambd, gen1);

  GenerateInputs(1, {{8, 3, 24, 24}});
  auto result2 = GetHpuInput(0).exponential_(lambd, gen2);

  EXPECT_TRUE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_inplace_bf16_5) {
  double lambd = GenerateScalar<double>(10.0, 50.0);

  GenerateInputs(1, {{8, 3, 24, 32, 32}}, {torch::kBFloat16});
  auto result1 = GetHpuInput(0).exponential_(lambd);

  GenerateInputs(1, {{8, 3, 24, 32, 32}}, {torch::kBFloat16});
  auto result2 = GetHpuInput(0).exponential_(lambd);

  EXPECT_TRUE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_f32_diff_seed) {
  double lambd = GenerateScalar<double>(1.0, 5.0);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/41216728023107);

  GenerateInputs(1, {{1024}});
  auto result1 = torch::exponential(GetHpuInput(0), lambd, gen1);

  GenerateInputs(1, {{1024}});
  auto result2 = torch::exponential(GetHpuInput(0), lambd, gen2);

  EXPECT_FALSE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_f32_2) {
  double lambd = GenerateScalar<double>(5.0, 15.0);

  GenerateInputs(1, {{256, 256}});
  auto result1 = torch::exponential(GetHpuInput(0), lambd);

  GenerateInputs(1, {{256, 256}});
  auto result2 = torch::exponential(GetHpuInput(0), lambd);

  EXPECT_TRUE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_bf16_diff_seed) {
  double lambd = GenerateScalar<double>(1.0, 5.0);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/41216728023107);

  GenerateInputs(1, {{1024}}, torch::kBFloat16);
  auto result1 = torch::exponential(GetHpuInput(0), lambd, gen1);

  GenerateInputs(1, {{1024}}, torch::kBFloat16);
  auto result2 = torch::exponential(GetHpuInput(0), lambd, gen2);

  EXPECT_FALSE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_bf16_2) {
  double lambd = GenerateScalar<double>(5.0, 15.0);

  GenerateInputs(1, {{256, 256}}, torch::kBFloat16);
  auto result1 = torch::exponential(GetHpuInput(0), lambd);

  GenerateInputs(1, {{256, 256}}, torch::kBFloat16);
  auto result2 = torch::exponential(GetHpuInput(0), lambd);

  EXPECT_TRUE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_out_f32_diff_seed) {
  double lambd = GenerateScalar<double>(1.0, 5.0);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/41216728023107);

  GenerateInputs(1, {{1024}});
  auto result1 = torch::empty(0, torch::kFloat32).to(torch::kHPU);
  torch::exponential_outf(GetHpuInput(0), lambd, gen1, result1);

  GenerateInputs(1, {{1024}});
  auto result2 = torch::empty(0, torch::kFloat32).to(torch::kHPU);
  torch::exponential_outf(GetHpuInput(0), lambd, gen2, result2);

  EXPECT_FALSE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_out_bf16_diff_seed) {
  double lambd = GenerateScalar<double>(1.0, 5.0);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/41216728023107);

  GenerateInputs(1, {{1024}}, torch::kBFloat16);
  auto result1 = torch::empty(0, torch::kBFloat16).to(torch::kHPU);
  torch::exponential_outf(GetHpuInput(0), lambd, gen1, result1);

  GenerateInputs(1, {{1024}}, torch::kBFloat16);
  auto result2 = torch::empty(0, torch::kBFloat16).to(torch::kHPU);
  torch::exponential_outf(GetHpuInput(0), lambd, gen2, result2);

  EXPECT_FALSE(torch::equal(result1, result2));
}

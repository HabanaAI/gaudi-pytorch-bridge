/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
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

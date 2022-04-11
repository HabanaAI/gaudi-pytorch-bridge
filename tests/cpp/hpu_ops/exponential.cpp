/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, exponential_f32_1) {
  double lambd = GenerateScalar<double>(1.0, 5.0);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);

  GenerateInputs(1, {{1024}});
  auto result1 = GetHpuInput(0).exponential_(lambd, gen1);

  GenerateInputs(1, {{1024}});
  auto result2 = GetHpuInput(0).exponential_(lambd, gen2);

  EXPECT_TRUE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_f32_diff_seed) {
  double lambd = GenerateScalar<double>(1.0, 5.0);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/41216728023107);

  GenerateInputs(1, {{1024}});
  auto result1 = GetHpuInput(0).exponential_(lambd, gen1);

  GenerateInputs(1, {{1024}});
  auto result2 = GetHpuInput(0).exponential_(lambd, gen2);

  EXPECT_FALSE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_f32_2) {
  double lambd = GenerateScalar<double>(5.0, 15.0);

  GenerateInputs(1, {{256, 256}});
  auto result1 = GetHpuInput(0).exponential_(lambd);

  GenerateInputs(1, {{256, 256}});
  auto result2 = GetHpuInput(0).exponential_(lambd);

  EXPECT_TRUE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_bf16_3) {
  double lambd = GenerateScalar<double>(3.0, 10.0);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/41216728023107);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/41216728023107);

  GenerateInputs(1, {{24, 32, 32}}, {torch::kBFloat16});
  auto result1 = GetHpuInput(0).exponential_(lambd, gen1);

  GenerateInputs(1, {{24, 32, 32}}, {torch::kBFloat16});
  auto result2 = GetHpuInput(0).exponential_(lambd, gen2);

  EXPECT_TRUE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_f32_4) {
  double lambd = GenerateScalar<double>(30.0, 80.0);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/16741280223107);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/16741280223107);

  GenerateInputs(1, {{8, 3, 24, 24}});
  auto result1 = GetHpuInput(0).exponential_(lambd, gen1);

  GenerateInputs(1, {{8, 3, 24, 24}});
  auto result2 = GetHpuInput(0).exponential_(lambd, gen2);

  EXPECT_TRUE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, exponential_bf16_5) {
  double lambd = GenerateScalar<double>(10.0, 50.0);

  GenerateInputs(1, {{8, 3, 24, 32, 32}}, {torch::kBFloat16});
  auto result1 = GetHpuInput(0).exponential_(lambd);

  GenerateInputs(1, {{8, 3, 24, 32, 32}}, {torch::kBFloat16});
  auto result2 = GetHpuInput(0).exponential_(lambd);

  EXPECT_TRUE(torch::equal(result1, result2));
}

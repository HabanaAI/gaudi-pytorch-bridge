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

TEST_F(HpuOpTest, geometric_f32) {
  GenerateInputs(2);
  double p = 0.8;

  auto result1 = GetHpuInput(0).geometric_(p);
  auto result2 = GetHpuInput(1).geometric_(p);

  EXPECT_FALSE(result1.equal(result2));

  torch::manual_seed(31);
  result1 =
      GetHpuInput(0).geometric_(p, at::detail::getDefaultCPUGenerator()).cpu();
  torch::manual_seed(31);
  result2 =
      GetHpuInput(1).geometric_(p, at::detail::getDefaultCPUGenerator()).cpu();

  EXPECT_TRUE(result1.equal(result2));
}

TEST_F(HpuOpTest, geometric_bf16) {
  GenerateInputs(2, torch::kBFloat16);
  double p = 0.9;

  auto result1 = GetHpuInput(0).geometric_(p);
  auto result2 = GetHpuInput(1).geometric_(p);

  EXPECT_FALSE(result1.equal(result2));

  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);

  result1 = GetHpuInput(0).geometric_(p, gen1).cpu();
  result2 = GetHpuInput(1).geometric_(p, gen2).cpu();

  EXPECT_TRUE(result1.equal(result2));
}

TEST_F(HpuOpTest, geometric_out_f32) {
  double p = GenerateScalar<double>(0.0, 1.0);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/41216728023107);

  GenerateInputs(1);
  auto result1 = torch::empty(0).to(torch::kHPU);
  torch::geometric_outf(GetHpuInput(0), p, gen1, result1);

  GenerateInputs(1);
  auto result2 = torch::empty(0).to(torch::kHPU);
  torch::geometric_outf(GetHpuInput(0), p, gen2, result2);

  EXPECT_FALSE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, geometric_out_bf16) {
  double p = GenerateScalar<double>(0.0, 1.0);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/41216728023107);

  GenerateInputs(1, torch::kBFloat16);
  auto result1 = torch::empty(0, torch::kBFloat16).to(torch::kHPU);
  torch::geometric_outf(GetHpuInput(0), p, gen1, result1);

  GenerateInputs(1, torch::kBFloat16);
  auto result2 = torch::empty(0, torch::kBFloat16).to(torch::kHPU);
  torch::geometric_outf(GetHpuInput(0), p, gen2, result2);

  EXPECT_FALSE(torch::equal(result1, result2));
}
TEST_F(HpuOpTest, geometric_out_f32_seed) {
  double p = GenerateScalar<double>(0.0, 1.0);

  GenerateInputs(1);
  auto result1 = torch::empty(0).to(torch::kHPU);
  SetSeed();
  torch::geometric_outf(
      GetHpuInput(0), p, at::detail::getDefaultCPUGenerator(), result1);

  GenerateInputs(1);
  auto result2 = torch::empty(0).to(torch::kHPU);
  SetSeed();
  torch::geometric_outf(
      GetHpuInput(0), p, at::detail::getDefaultCPUGenerator(), result1);

  EXPECT_FALSE(torch::equal(result1, result2));
}

TEST_F(HpuOpTest, geometric_out_bf16_seed) {
  double p = GenerateScalar<double>(0.0, 1.0);

  GenerateInputs(1, torch::kBFloat16);
  auto result1 = torch::empty(0, torch::kBFloat16).to(torch::kHPU);
  SetSeed();
  torch::geometric_outf(
      GetHpuInput(0), p, at::detail::getDefaultCPUGenerator(), result1);

  GenerateInputs(1, torch::kBFloat16);
  auto result2 = torch::empty(0, torch::kBFloat16).to(torch::kHPU);
  SetSeed();
  torch::geometric_outf(
      GetHpuInput(0), p, at::detail::getDefaultCPUGenerator(), result1);
  EXPECT_FALSE(torch::equal(result1, result2));
}

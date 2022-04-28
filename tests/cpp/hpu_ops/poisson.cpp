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
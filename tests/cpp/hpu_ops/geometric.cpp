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

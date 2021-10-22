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

TEST_F(HpuOpTest, random_) {
  GenerateInputs(1);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  GenerateInputs(1);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  Compare(result1, result2);
}

TEST_F(HpuOpTest, random_from) {
  GenerateInputs(1);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(9, 10);

  GenerateInputs(1);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(9, 10);

  Compare(result1, result2);
  EXPECT_TRUE(result1.cpu().ge(9).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().lt(10).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_to) {
  GenerateInputs(1, torch::kInt);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(1000);

  GenerateInputs(1, torch::kInt);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(1000);

  Compare(result1.cpu(), result2);
  EXPECT_TRUE(result1.cpu().lt(1000).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, multinomial) {
  GenerateInputs(1, {{6, 8}}, torch::kFloat);
  auto c_sample = 2;
  SetSeed();
  auto result1 = torch::multinomial(GetHpuInput(0), c_sample);
  auto result2 = torch::multinomial(GetHpuInput(0), c_sample);

  Compare(result1, result2);
}

TEST_F(HpuOpTest, multinomial_replacement) {
  GenerateInputs(1, {{5, 80}}, torch::kFloat);
  auto c_sample = 4;
  SetSeed();
  auto result1 = torch::multinomial(GetHpuInput(0), c_sample, true);
  auto result2 = torch::multinomial(GetHpuInput(0), c_sample, true);

  Compare(result1, result2);
}

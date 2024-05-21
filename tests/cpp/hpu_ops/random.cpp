/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

#include "../utils/device_type_util.h"
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, uniform_) {
  GenerateInputs(2);

  auto result1 = GetHpuInput(0).uniform_().cpu();
  auto result2 = GetHpuInput(1).uniform_().cpu();

  EXPECT_FALSE(result1.equal(result2));

  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto from = GenerateScalar<float>(0.3, 0.5);
  auto to = GenerateScalar<float>(0.6, 0.7);

  result1 = GetHpuInput(0).uniform_(from, to, gen1).cpu();
  result2 = GetHpuInput(1).uniform_(from, to, gen2).cpu();

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.ge(from).all().item().toBool());
  EXPECT_TRUE(result1.lt(to).all().item().toBool());
}

TEST_F(HpuOpTest, normal_) {
  GenerateInputs(2);

  auto result1 = GetHpuInput(0).normal_().cpu();
  auto result2 = GetHpuInput(1).normal_().cpu();

  EXPECT_FALSE(result1.equal(result2));

  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto mean = GenerateScalar<float>();
  auto std = GenerateScalar<float>();
  GenerateInputs(2, torch::kBFloat16);
  result1 = GetHpuInput(0).normal_(mean, std, gen1).cpu();
  result2 = GetHpuInput(1).normal_(mean, std, gen2).cpu();
  EXPECT_TRUE(result1.equal(result2));
}

TEST_F(HpuOpTest, log_normal_) {
  GenerateInputs(2);

  auto result1 = GetHpuInput(0).log_normal_().cpu();
  auto result2 = GetHpuInput(1).log_normal_().cpu();

  EXPECT_FALSE(result1.equal(result2));

  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto mean = GenerateScalar<float>();
  auto std = GenerateScalar<float>();
  GenerateInputs(2, torch::kBFloat16);
  result1 = GetHpuInput(0).log_normal_(mean, std, gen1).cpu();
  result2 = GetHpuInput(1).log_normal_(mean, std, gen2).cpu();
  EXPECT_TRUE(result1.equal(result2));
}

TEST_F(HpuOpTest, bernoulli_) {
  GenerateInputs(3);

  auto result1 = GetHpuInput(0).bernoulli_().cpu();
  auto result2 = GetHpuInput(1).bernoulli_().cpu();

  EXPECT_FALSE(result1.equal(result2));

  auto p = GetHpuInput(2);
  torch::manual_seed(31);
  result1 =
      GetHpuInput(0).bernoulli_(p, at::detail::getDefaultCPUGenerator()).cpu();
  torch::manual_seed(31);
  result2 =
      GetHpuInput(1).bernoulli_(p, at::detail::getDefaultCPUGenerator()).cpu();

  EXPECT_TRUE(result1.equal(result2));
}

TEST_F(HpuOpTest, bernoulli) {
  GenerateInputs(3, torch::kBFloat16);

  auto result1 = torch::bernoulli(GetHpuInput(0)).cpu();
  auto result2 = torch::bernoulli(GetHpuInput(1)).cpu();

  EXPECT_FALSE(result1.equal(result2));

  auto p = GetHpuInput(2);
  SetSeed();
  result1 = torch::bernoulli(GetHpuInput(0)).cpu();
  SetSeed();
  result2 = torch::bernoulli(GetHpuInput(0)).cpu();

  EXPECT_TRUE(result1.equal(result2));
}

TEST_F(HpuOpTest, bernoulli_p) {
  GenerateInputs(3, torch::kBFloat16);

  auto result1 = torch::bernoulli(GetHpuInput(0), 0.75).cpu();
  auto result2 = torch::bernoulli(GetHpuInput(1), 0.75).cpu();

  EXPECT_FALSE(result1.equal(result2));

  auto p = GetHpuInput(2);
  SetSeed();
  result1 = torch::bernoulli(GetHpuInput(0), 0.42).cpu();
  SetSeed();
  result2 = torch::bernoulli(GetHpuInput(0), 0.42).cpu();

  EXPECT_TRUE(result1.equal(result2));
}

TEST_F(HpuOpTest, bernoulli_out) {
  GenerateInputs(1, {{64, 64}}, torch::kBFloat16);
  auto input = GetHpuInput(0);

  auto out1 =
      torch::empty(0, at::TensorOptions(torch::kBFloat16).device(at::kHPU));
  auto out2 =
      torch::empty(0, at::TensorOptions(torch::kBFloat16).device(at::kHPU));

  SetSeed();
  torch::bernoulli_outf(input, at::detail::getDefaultCPUGenerator(), out1);
  SetSeed();
  torch::bernoulli_outf(input, at::detail::getDefaultCPUGenerator(), out2);

  EXPECT_TRUE(out1.equal(out2));

  auto out3 =
      torch::empty(0, at::TensorOptions(torch::kBFloat16).device(at::kHPU));
  torch::bernoulli_outf(input, at::detail::getDefaultCPUGenerator(), out3);

  EXPECT_FALSE(out1.equal(out3));
}

TEST_F(HpuOpTest, bernoulli_out_2) {
  if (isGaudi()) {
    GTEST_SKIP() << "Test skipped on Gaudi.";
  }
  GenerateInputs(1, {{64, 64}}, torch::kHalf);
  auto input = GetHpuInput(0);

  auto out1 = torch::empty(0, at::TensorOptions(torch::kHalf).device(at::kHPU));
  auto out2 = torch::empty(0, at::TensorOptions(torch::kHalf).device(at::kHPU));

  SetSeed();
  torch::bernoulli_outf(input, at::detail::getDefaultCPUGenerator(), out1);
  SetSeed();
  torch::bernoulli_outf(input, at::detail::getDefaultCPUGenerator(), out2);

  EXPECT_TRUE(out1.equal(out2));

  auto out3 = torch::empty(0, at::TensorOptions(torch::kHalf).device(at::kHPU));
  torch::bernoulli_outf(input, at::detail::getDefaultCPUGenerator(), out3);

  EXPECT_FALSE(out1.equal(out3));
}

TEST_F(HpuOpTest, bernoulli_out_scalar1) {
  GenerateInputs(1, {{64, 64}}, torch::kBFloat16);
  auto input = GetHpuInput(0);

  auto out1 =
      torch::empty(0, at::TensorOptions(torch::kBFloat16).device(at::kHPU));
  auto out2 =
      torch::empty(0, at::TensorOptions(torch::kBFloat16).device(at::kHPU));

  SetSeed();
  torch::bernoulli_outf(input, 0.8, at::detail::getDefaultCPUGenerator(), out1);
  SetSeed();
  torch::bernoulli_outf(input, 0.8, at::detail::getDefaultCPUGenerator(), out2);

  EXPECT_TRUE(out1.equal(out2));

  auto out3 =
      torch::empty(0, at::TensorOptions(torch::kBFloat16).device(at::kHPU));
  torch::bernoulli_outf(input, 0.8, at::detail::getDefaultCPUGenerator(), out3);

  EXPECT_FALSE(out1.equal(out3));
}

TEST_F(HpuOpTest, bernoulli_out_scalar2) {
  if (isGaudi()) {
    GTEST_SKIP() << "Test skipped on Gaudi.";
  }
  GenerateInputs(1, {{64, 64}}, torch::kHalf);
  auto input = GetHpuInput(0);

  auto out1 = torch::empty(0, at::TensorOptions(torch::kHalf).device(at::kHPU));
  auto out2 = torch::empty(0, at::TensorOptions(torch::kHalf).device(at::kHPU));

  SetSeed();
  torch::bernoulli_outf(input, 0.3, at::detail::getDefaultCPUGenerator(), out1);
  SetSeed();
  torch::bernoulli_outf(input, 0.3, at::detail::getDefaultCPUGenerator(), out2);

  EXPECT_TRUE(out1.equal(out2));

  auto out3 = torch::empty(0, at::TensorOptions(torch::kHalf).device(at::kHPU));
  torch::bernoulli_outf(input, 0.3, at::detail::getDefaultCPUGenerator(), out3);

  EXPECT_FALSE(out1.equal(out3));
}

TEST_F(HpuOpTest, random_f32) {
  GenerateInputs(1, torch::kFloat32);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  GenerateInputs(1, torch::kFloat32);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.equal(result1.floor()));
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_from_f32) {
  GenerateInputs(1, torch::kFloat32);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(9, 10);

  GenerateInputs(1, torch::kFloat32);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(9, 10);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.equal(result1.floor()));
  EXPECT_TRUE(result1.cpu().ge(9).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().lt(10).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_to_f32) {
  GenerateInputs(1, torch::kFloat32);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(1000);

  GenerateInputs(1, torch::kFloat32);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(1000);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.equal(result1.floor()));
  EXPECT_TRUE(result1.cpu().lt(1 << 24).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_i32) {
  GenerateInputs(1, torch::kInt32);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  GenerateInputs(1, torch::kInt32);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.cpu()
                  .lt(std::numeric_limits<int32_t>::max())
                  .all()
                  .item()
                  .toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_from_i32) {
  GenerateInputs(1, torch::kInt32);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(-10, 10);

  GenerateInputs(1, torch::kInt32);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(-10, 10);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.cpu().ge(-10).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().lt(10).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_to_i32) {
  GenerateInputs(1, torch::kInt32);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(1000);

  GenerateInputs(1, torch::kInt32);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(1000);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.cpu().lt(1000).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_i16) {
  GenerateInputs(1, torch::kInt16);
  SetSeed();
  auto result1 = GetHpuInput(0)
                     .random_(at::detail::getDefaultCPUGenerator())
                     .to(torch::kInt32);

  GenerateInputs(1, torch::kInt16);
  SetSeed();
  auto result2 = GetHpuInput(0)
                     .random_(at::detail::getDefaultCPUGenerator())
                     .to(torch::kInt32);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.cpu()
                  .lt(std::numeric_limits<int16_t>::max())
                  .all()
                  .item()
                  .toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_from_i16) {
  GenerateInputs(1, torch::kInt16);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(-10, 10).to(torch::kInt32);

  GenerateInputs(1, torch::kInt16);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(-10, 10).to(torch::kInt32);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.cpu().ge(-10).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().lt(10).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_to_i16) {
  GenerateInputs(1, torch::kInt16);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(1000).to(torch::kInt32);

  GenerateInputs(1, torch::kInt16);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(1000).to(torch::kInt32);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.cpu().lt(1000).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_i8) {
  GenerateInputs(1, torch::kChar);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  GenerateInputs(1, torch::kChar);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.cpu().le(127).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_from_i8) {
  GenerateInputs(1, torch::kChar);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(-10, 10);

  GenerateInputs(1, torch::kChar);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(-10, 10);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.cpu().ge(-10).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().lt(10).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_to_i8) {
  GenerateInputs(1, torch::kChar);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(127);

  GenerateInputs(1, torch::kChar);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(127);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.cpu().lt(127).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_u8) {
  GenerateInputs(1, torch::kByte);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  GenerateInputs(1, torch::kByte);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.cpu().le(255).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_from_u8) {
  GenerateInputs(1, torch::kByte);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(0, 20);

  GenerateInputs(1, torch::kByte);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(0, 20);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().lt(20).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_to_u8) {
  GenerateInputs(1, torch::kByte);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(255);

  GenerateInputs(1, torch::kByte);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(255);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.cpu().lt(255).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_half) {
  if (isGaudi()) {
    GTEST_SKIP() << "Test skipped on Gaudi.";
  }
  GenerateInputs(1, torch::kHalf);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  GenerateInputs(1, torch::kHalf);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.equal(result1.floor()));
  EXPECT_TRUE(result1.cpu().le((1 << 11)).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_from_half) {
  if (isGaudi()) {
    GTEST_SKIP() << "Test skipped on Gaudi.";
  }
  GenerateInputs(1, torch::kHalf);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(-10, 10);

  GenerateInputs(1, torch::kHalf);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(-10, 10);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.equal(result1.floor()));
  EXPECT_TRUE(result1.cpu().ge(-10).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().lt(10).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_to_half) {
  if (isGaudi()) {
    GTEST_SKIP() << "Test skipped on Gaudi.";
  }
  GenerateInputs(1, torch::kHalf);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(1000);

  GenerateInputs(1, torch::kHalf);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(1000);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.equal(result1.floor()));
  EXPECT_TRUE(result1.cpu().lt(1000).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_bf16) {
  GenerateInputs(1, torch::kBFloat16);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  GenerateInputs(1, torch::kBFloat16);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(at::detail::getDefaultCPUGenerator());

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.equal(result1.floor()));
  EXPECT_TRUE(result1.cpu().le(255).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_from_bf16) {
  GenerateInputs(1, torch::kBFloat16);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(-10, 10);

  GenerateInputs(1, torch::kBFloat16);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(-10, 10);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.equal(result1.floor()));
  EXPECT_TRUE(result1.cpu().ge(-10).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().lt(10).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, random_to_bf16) {
  GenerateInputs(1, torch::kBFloat16);
  SetSeed();
  auto result1 = GetHpuInput(0).random_(127);

  GenerateInputs(1, torch::kBFloat16);
  SetSeed();
  auto result2 = GetHpuInput(0).random_(127);

  EXPECT_TRUE(result1.equal(result2));
  EXPECT_TRUE(result1.equal(result1.floor()));
  EXPECT_TRUE(result1.cpu().lt(127).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
  EXPECT_TRUE(result1.cpu().ge(0).all().item().toBool())
      << "Seed=" << GetSeed() << "\n";
}

TEST_F(HpuOpTest, multinomial) {
  GenerateInputs(1, {{64, 64}});
  auto c_sample = 2;
  SetSeed();
  auto result1 = torch::multinomial(GetHpuInput(0), c_sample);
  SetSeed();
  auto result2 = torch::multinomial(GetHpuInput(0), c_sample);

  Compare(result1.cpu(), result2);
}

TEST_F(HpuOpTest, multinomial_without_replacement) {
  GenerateInputs(1, {{64, 64}});
  auto c_sample = 4;
  SetSeed();
  auto result1 = torch::multinomial(GetHpuInput(0), c_sample, false);
  SetSeed();
  auto result2 = torch::multinomial(GetHpuInput(0), c_sample, false);

  Compare(result1.cpu(), result2);
}

TEST_F(HpuOpTest, multinomial_with_replacement) {
  GenerateInputs(1, {{64, 64}});
  auto c_sample = 4;
  SetSeed();
  auto result1 = torch::multinomial(GetHpuInput(0), c_sample, true);
  SetSeed();
  auto result2 = torch::multinomial(GetHpuInput(0), c_sample, true);

  Compare(result1.cpu(), result2);
}

// Expected hpu results are different for different seed run
TEST_F(HpuOpTest, multinomial_with_replacement_and_different_seed) {
  GenerateInputs(1, {{64, 64}});
  auto c_sample = 4;
  torch::manual_seed(10);
  auto result1 = torch::multinomial(GetHpuInput(0), c_sample, true);
  torch::manual_seed(20);
  auto result2 = torch::multinomial(GetHpuInput(0), c_sample, true);
  EXPECT_FALSE(result1.equal(result2));
}

// Expected hpu results are different for different seed run
TEST_F(HpuOpTest, multinomial_without_replacement_and_different_seed) {
  GenerateInputs(1, {{64, 64}});
  auto c_sample = 4;
  torch::manual_seed(10);
  auto result1 = torch::multinomial(GetHpuInput(0), c_sample, false);
  torch::manual_seed(20);
  auto result2 = torch::multinomial(GetHpuInput(0), c_sample, false);
  EXPECT_FALSE(result1.equal(result2));
}

TEST_F(HpuOpTest, multinomial_out) {
  GenerateInputs(1, {{64, 64}});
  auto c_sample = 2;
  auto result = torch::empty(0, torch::kInt).to(torch::kHPU);
  SetSeed();
  auto result1 = torch::multinomial_outf(
      GetHpuInput(0),
      c_sample,
      false,
      at::detail::getDefaultCPUGenerator(),
      result);
  SetSeed();
  auto result2 = torch::multinomial_outf(
      GetHpuInput(0),
      c_sample,
      false,
      at::detail::getDefaultCPUGenerator(),
      result);
  Compare(result1.cpu(), result2);
}

TEST_F(HpuOpTest, multinomial_out_with_replacement) {
  GenerateInputs(1, {{64, 64}});
  auto c_sample = 2;
  auto result = torch::empty(0, torch::kInt).to(torch::kHPU);
  SetSeed();
  auto result1 = torch::multinomial_outf(
      GetHpuInput(0),
      c_sample,
      true,
      at::detail::getDefaultCPUGenerator(),
      result);
  SetSeed();
  auto result2 = torch::multinomial_outf(
      GetHpuInput(0),
      c_sample,
      true,
      at::detail::getDefaultCPUGenerator(),
      result);
  Compare(result1.cpu(), result2);
}

// Expected hpu results are different for different seed run
TEST_F(HpuOpTest, multinomial_out_with_replacement_different_seed) {
  GenerateInputs(1, {{64, 64}});
  auto c_sample = 4;
  auto out1 = torch::empty(0, torch::kInt).to(torch::kHPU);
  auto out2 = torch::empty(0, torch::kInt).to(torch::kHPU);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/41216728023107);
  auto result1 =
      torch::multinomial_outf(GetHpuInput(0), c_sample, true, gen1, out1);
  auto result2 =
      torch::multinomial_outf(GetHpuInput(0), c_sample, true, gen2, out2);
  EXPECT_FALSE(result1.equal(result2));
}

// Expected hpu results are different for different seed run
TEST_F(HpuOpTest, multinomial_out_without_replacement_different_seed) {
  GenerateInputs(1, {{64, 64}});
  auto c_sample = 4;
  auto out1 = torch::empty(0, torch::kInt).to(torch::kHPU);
  auto out2 = torch::empty(0, torch::kInt).to(torch::kHPU);
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/41216728023107);
  auto result1 =
      torch::multinomial_outf(GetHpuInput(0), c_sample, false, gen1, out1);
  auto result2 =
      torch::multinomial_outf(GetHpuInput(0), c_sample, false, gen2, out2);
  EXPECT_FALSE(result1.equal(result2));
}

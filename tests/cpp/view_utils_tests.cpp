/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

#include <gtest/gtest.h>
#include <torch/torch.h>

TEST(ViewUtilsTest, IsAliasSameTensor) {
  auto cpu_a = torch::ones(5);
  auto hpu_a = cpu_a.to(torch::kHPU);
  auto cpu_b = cpu_a;
  auto hpu_b = hpu_a;
  auto cpu_out = cpu_a.is_alias_of(cpu_b);
  auto hpu_out = hpu_a.is_alias_of(hpu_b);
  EXPECT_TRUE(cpu_out == hpu_out);
}

TEST(ViewUtilsTest, IsAliasAsStrided) {
  auto cpu_a = torch::ones(5);

  auto hpu_a = cpu_a.to(torch::kHPU);

  auto cpu_a_as_strided = cpu_a.as_strided(2, 2);
  auto hpu_a_as_strided = hpu_a.as_strided(2, 2);

  auto cpu_out = cpu_a.is_alias_of(cpu_a_as_strided);
  auto hpu_out = hpu_a.is_alias_of(hpu_a_as_strided);
  EXPECT_TRUE(cpu_out == hpu_out);
}

TEST(ViewUtilsTest, IsAliasAsStridedMulOut) {
  auto cpu_a = torch::ones(5);
  auto cpu_b = torch::ones(2);
  auto cpu_c = torch::ones(2);

  auto hpu_a = cpu_a.to(torch::kHPU);
  auto hpu_b = cpu_b.to(torch::kHPU);
  auto hpu_c = cpu_c.to(torch::kHPU);

  auto cpu_a_as_strided = cpu_a.as_strided(2, 2);
  auto hpu_a_as_strided = hpu_a.as_strided(2, 2);

  mul_out(cpu_a_as_strided, cpu_b, cpu_c);
  mul_out(hpu_a_as_strided, hpu_b, hpu_c);

  auto cpu_out = cpu_a.is_alias_of(cpu_a_as_strided);
  auto hpu_out = hpu_a.is_alias_of(hpu_a_as_strided);
  EXPECT_TRUE(cpu_out == hpu_out);
}

TEST(ViewUtilsTest, IsAliasAsStridedMulOutAsStrided) {
  auto cpu_a = torch::ones(5);
  auto cpu_b = torch::ones(2);
  auto cpu_c = torch::ones(2);

  auto hpu_a = cpu_a.to(torch::kHPU);
  auto hpu_b = cpu_b.to(torch::kHPU);
  auto hpu_c = cpu_c.to(torch::kHPU);

  auto cpu_a_as_strided = cpu_a.as_strided(2, 2);
  auto hpu_a_as_strided = hpu_a.as_strided(2, 2);

  mul_out(cpu_a_as_strided, cpu_b, cpu_c);
  mul_out(cpu_a_as_strided, cpu_b, cpu_c);
  mul_out(hpu_a_as_strided, hpu_b, hpu_c);
  mul_out(hpu_a_as_strided, hpu_b, hpu_c);

  auto cpu_a1_as_strided = cpu_a_as_strided.as_strided(1, 1);
  auto hpu_a1_as_strided = hpu_a_as_strided.as_strided(1, 1);

  auto cpu_out = cpu_a.is_alias_of(cpu_a1_as_strided);
  auto hpu_out = hpu_a.is_alias_of(hpu_a1_as_strided);
  EXPECT_TRUE(cpu_out == hpu_out);
}

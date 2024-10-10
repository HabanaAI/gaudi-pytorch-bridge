/*******************************************************************************
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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "backend/helpers/dynamic_shape_info.h"
#include "habana_lazy/hpu_lazy_tensors.h"

using namespace habana_lazy;

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
  auto cpu_a1_as_strided = cpu_a_as_strided.view(-1);
  auto hpu_a1_as_strided = hpu_a_as_strided.view(-1);

  auto cpu_out = cpu_a1_as_strided.is_alias_of(cpu_a_as_strided);
  auto hpu_out = hpu_a1_as_strided.is_alias_of(hpu_a_as_strided);
  EXPECT_TRUE(cpu_out == hpu_out);

  cpu_out = cpu_a.is_alias_of(cpu_a_as_strided);
  hpu_out = hpu_a.is_alias_of(hpu_a_as_strided);
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

TEST(ViewUtilsTest, IsAliasViews) {
  auto cpu_a = torch::ones(5);
  auto hpu_a = cpu_a.to(torch::kHPU); // This will contain internal at tensor

  auto hpu_b = hpu_a.view(-1); // This will contain frontend at tensor
  auto hpu_c = hpu_b.view(-1);

  // Frontend tensor is_alias_of backend tensor
  auto hpu_out = hpu_b.is_alias_of(hpu_a);
  EXPECT_TRUE(true == hpu_out);

  // Frontend tensor is_alias_of frontend tensor
  hpu_out = hpu_c.is_alias_of(hpu_b);
  EXPECT_TRUE(true == hpu_out);

  // Backend tensor is_alias_of frontend tensor
  hpu_out = hpu_a.is_alias_of(hpu_c);
  EXPECT_TRUE(true == hpu_out);
}

TEST(ViewUtilsTest, IsAliasNonZeroOpViews) {
  torch::Tensor input_cpu = torch::tensor({});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
  auto hpu_nz = torch::nonzero(input_hpu);
  auto cpu_nz = torch::nonzero(input_cpu);
  auto hpu_view = hpu_nz.view(-1);
  auto cpu_view = cpu_nz.view(-1);

  auto hpu_out = hpu_nz.is_alias_of(hpu_view);
  auto cpu_out = cpu_nz.is_alias_of(cpu_view);
  EXPECT_TRUE(cpu_out == hpu_out);

  hpu_out = hpu_view.is_alias_of(hpu_nz);
  cpu_out = cpu_view.is_alias_of(cpu_nz);
  EXPECT_TRUE(cpu_out == hpu_out);
}

TEST(ViewUtilsTest, IsAliasSliceOnChlastInput) {
  bool refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  if (!refine_enabled) {
    habana_helpers::EnableRefineDynamicShape();
  }

  int N = 2, C = 3, H = 4, W = 5;
  std::vector<int> in_sizes{8, 10, 12, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    torch::Tensor A =
        torch::randn({N, C, H, W}).contiguous(c10::MemoryFormat::ChannelsLast);
    auto hA = A.to(torch::kHPU);
    auto B = torch::slice(A, 1, 1, -1, 1);
    auto hB = torch::slice(hA, 1, 1, -1, 1);

    auto cpu_out = A.is_alias_of(B);
    auto hpu_out = hA.is_alias_of(hB);
    HbLazyTensor::StepMarker({});
    EXPECT_EQ(allclose(B, hB.cpu()), true);
    EXPECT_TRUE(cpu_out == hpu_out);
  }

  if (!refine_enabled) {
    habana_helpers::DisableRefineDynamicShape();
  }
}
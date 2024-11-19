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

#include <limits>
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, nansum_4d_2d_keepdim) {
  GenerateInputs(1, {{2, 3, 4, 5}});
  const std::vector<int64_t> dim{0, 2};

  auto expected = torch::nansum(GetCpuInput(0), dim, true /*keepdim*/);
  auto result = torch::nansum(GetHpuInput(0), dim, true /*keepdim*/);

  Compare(expected, result);
}

TEST_F(HpuOpTest, nansum_3d_2d_keepdim_out) {
  GenerateInputs(1, {{5, 3, 6}});
  const std::vector<int64_t> dim{-2, 0};
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({1, 1, 6}, dtype);
  auto result =
      torch::empty({1, 1, 6}, torch::TensorOptions(dtype).device("hpu"));

  torch::nansum_outf(GetCpuInput(0), dim, true /*keepdim*/, dtype, expected);
  torch::nansum_outf(GetHpuInput(0), dim, true /*keepdim*/, dtype, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, nansum_4d_4d_reduce_dim_out) {
  GenerateInputs(1, {{4, 6, 3, 2}});
  const std::vector<int64_t> dim{2, 1, -4, -1};
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::nansum_outf(GetCpuInput(0), dim, false /*keepdim*/, dtype, expected);
  torch::nansum_outf(GetHpuInput(0), dim, false /*keepdim*/, dtype, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, nansum_4d_3d_reduce_dim) {
  GenerateInputs(1, {{3, 6, 5, 4}});
  const std::vector<int64_t> dim{3, 1, 0};

  auto expected = torch::nansum(GetCpuInput(0), dim, false /*keepdim*/);
  auto result = torch::nansum(GetHpuInput(0), dim, false /*keepdim*/);

  Compare(expected, result);
}

TEST_F(HpuOpTest, nansum) {
  GenerateInputs(1, {{3, 3, 4, 6}});

  auto expected = torch::nansum(GetCpuInput(0));
  auto result = torch::nansum(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, nansum_f32) {
  auto t1_cpu = torch::tensor({1, 2}, torch::kF32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);

  auto out_cpu = torch::nansum(t1_cpu, 0, false);
  auto out_hpu = torch::nansum(t1_hpu, 0, false);

  EXPECT_TRUE(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001));
  EXPECT_TRUE(out_cpu.dtype() == out_hpu.dtype());
}

TEST_F(HpuOpTest, nansum_f32_with_nan) {
  auto t1_cpu = torch::tensor(
      {1.f, 2.f, std::numeric_limits<float>::quiet_NaN()}, torch::kF32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);

  auto out_cpu = torch::nansum(t1_cpu, 0, false);
  auto out_hpu = torch::nansum(t1_hpu, 0, false);

  EXPECT_TRUE(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001));
  EXPECT_TRUE(out_cpu.dtype() == out_hpu.dtype());
}

TEST_F(HpuOpTest, nansum_f32_dty_bf16) {
  auto t1_cpu = torch::tensor({1, 2}, torch::kF32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);

  auto out_cpu = torch::nansum(t1_cpu, 0, false, torch::kBFloat16);
  auto out_hpu = torch::nansum(t1_hpu, 0, false, torch::kBFloat16);

  EXPECT_TRUE(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001));
  EXPECT_TRUE(out_cpu.dtype() == out_hpu.dtype());
}

TEST_F(HpuOpTest, nansum_f32_with_nan_dty_bf16) {
  auto t1_cpu = torch::tensor(
      {1.f, 2.f, std::numeric_limits<float>::quiet_NaN()}, torch::kF32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);

  auto out_cpu = torch::nansum(t1_cpu, 0, false, torch::kBFloat16);
  auto out_hpu = torch::nansum(t1_hpu, 0, false, torch::kBFloat16);

  EXPECT_TRUE(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001));
  EXPECT_TRUE(out_cpu.dtype() == out_hpu.dtype());
}

// Put nanmean tests here because nanmean underneath uses nansum
TEST_F(HpuOpTest, nanmean_f32) {
  auto t1_cpu = torch::tensor({1, 2}, torch::kF32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);

  auto out_cpu = torch::nanmean(t1_cpu, 0, false);
  auto out_hpu = torch::nanmean(t1_hpu, 0, false);

  EXPECT_TRUE(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001));
  EXPECT_TRUE(out_cpu.dtype() == out_hpu.dtype());
}

TEST_F(HpuOpTest, nanmean_f32_with_nan) {
  auto t1_cpu = torch::tensor(
      {1.f, 2.f, std::numeric_limits<float>::quiet_NaN()}, torch::kF32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);

  auto out_cpu = torch::nanmean(t1_cpu, 0, false);
  auto out_hpu = torch::nanmean(t1_hpu, 0, false);

  EXPECT_TRUE(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001));
  EXPECT_TRUE(out_cpu.dtype() == out_hpu.dtype());
}

TEST_F(HpuOpTest, nanmean_f32_dty_bf16) {
  auto t1_cpu = torch::tensor({1, 2}, torch::kF32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);

  auto out_cpu = torch::nanmean(t1_cpu, 0, false, torch::kBFloat16);
  auto out_hpu = torch::nanmean(t1_hpu, 0, false, torch::kBFloat16);

  EXPECT_TRUE(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001));
  EXPECT_TRUE(out_cpu.dtype() == out_hpu.dtype());
}

TEST_F(HpuOpTest, nanmean_f32_with_nan_dty_bf16) {
  auto t1_cpu = torch::tensor(
      {1.f, 2.f, std::numeric_limits<float>::quiet_NaN()}, torch::kF32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);

  auto out_cpu = torch::nanmean(t1_cpu, 0, false, torch::kBFloat16);
  auto out_hpu = torch::nanmean(t1_hpu, 0, false, torch::kBFloat16);

  EXPECT_TRUE(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001));
  EXPECT_TRUE(out_cpu.dtype() == out_hpu.dtype());
}
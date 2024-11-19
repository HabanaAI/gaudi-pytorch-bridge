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

#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;
using namespace at;

class TypePromotionTests : public habana_lazy_test::LazyTest {};

TEST_F(TypePromotionTests, InplaceAdd) {
  auto t1_cpu = torch::tensor({22.0}, torch::kBFloat16);
  auto t2_cpu = torch::tensor({22}, torch::kInt32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  t1_cpu.add_(t2_cpu);
  t1_hpu.add_(t2_hpu);

  EXPECT_EQ(allclose(t1_hpu.to(torch::kCPU), t1_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t1_hpu.dtype() == t1_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, InplaceSub) {
  auto t1_cpu = torch::tensor({22.0}, torch::kBFloat16);
  auto t2_cpu = torch::tensor({22}, torch::kInt32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  t1_cpu.sub_(t2_cpu);
  t1_hpu.sub_(t2_hpu);

  EXPECT_EQ(allclose(t1_hpu.to(torch::kCPU), t1_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t1_hpu.dtype() == t1_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, InplaceMul) {
  auto t1_cpu = torch::tensor({22.0}, torch::kBFloat16);
  auto t2_cpu = torch::tensor({22}, torch::kInt32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  t1_cpu.mul_(t2_cpu);
  t1_hpu.mul_(t2_hpu);

  EXPECT_EQ(allclose(t1_hpu.to(torch::kCPU), t1_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t1_hpu.dtype() == t1_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, OutAdd) {
  auto t1_cpu = torch::tensor({22.0}, torch::kInt32);
  auto t2_cpu = torch::tensor({22.0}, torch::kBFloat16);
  auto t3_cpu = torch::tensor({22.0}, torch::kFloat);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  auto t3_hpu = t3_cpu.to(torch::kHPU);
  torch::add_out(t3_cpu, t1_cpu, t2_cpu);
  torch::add_out(t3_hpu, t1_hpu, t2_hpu);

  EXPECT_EQ(allclose(t3_hpu.to(torch::kCPU), t3_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t3_hpu.dtype() == t3_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, OutSub) {
  auto t1_cpu = torch::tensor({22.0}, torch::kInt32);
  auto t2_cpu = torch::tensor({22.0}, torch::kBFloat16);
  auto t3_cpu = torch::tensor({22.0}, torch::kFloat);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  auto t3_hpu = t3_cpu.to(torch::kHPU);
  torch::sub_out(t3_cpu, t1_cpu, t2_cpu);
  torch::sub_out(t3_hpu, t1_hpu, t2_hpu);

  EXPECT_EQ(allclose(t3_hpu.to(torch::kCPU), t3_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t3_hpu.dtype() == t3_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, OutMul) {
  auto t1_cpu = torch::tensor({22.0}, torch::kInt32);
  auto t2_cpu = torch::tensor({22.0}, torch::kBFloat16);
  auto t3_cpu = torch::tensor({22.0}, torch::kFloat);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  auto t3_hpu = t3_cpu.to(torch::kHPU);
  torch::mul_out(t3_cpu, t1_cpu, t2_cpu);
  torch::mul_out(t3_hpu, t1_hpu, t2_hpu);

  EXPECT_EQ(allclose(t3_hpu.to(torch::kCPU), t3_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t3_hpu.dtype() == t3_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, SharedInputMul) {
  auto t1_cpu = torch::tensor({22.0}, torch::kInt32);
  auto t2_cpu = torch::tensor({22.0}, torch::kFloat);
  auto t3_cpu = torch::tensor({22.0}, torch::kFloat);
  auto t4_cpu = torch::mul(t1_cpu, t2_cpu);
  auto t5_cpu = torch::mul(t1_cpu, t3_cpu);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  auto t3_hpu = t3_cpu.to(torch::kHPU);
  auto t4_hpu = torch::mul(t1_hpu, t2_hpu);
  auto t5_hpu = torch::mul(t1_hpu, t3_hpu);

  EXPECT_EQ(allclose(t4_hpu.to(torch::kCPU), t4_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t4_hpu.dtype() == t4_cpu.dtype(), true);

  EXPECT_EQ(allclose(t5_hpu.to(torch::kCPU), t5_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t5_hpu.dtype() == t5_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, SharedInputSub) {
  auto t1_cpu = torch::tensor({22.0}, torch::kInt32);
  auto t2_cpu = torch::tensor({22.0}, torch::kFloat);
  auto t3_cpu = torch::tensor({22.0}, torch::kFloat);
  auto t4_cpu = torch::sub(t1_cpu, t2_cpu);
  auto t5_cpu = torch::sub(t1_cpu, t3_cpu);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  auto t3_hpu = t3_cpu.to(torch::kHPU);
  auto t4_hpu = torch::sub(t1_hpu, t2_hpu);
  auto t5_hpu = torch::sub(t1_hpu, t3_hpu);

  EXPECT_EQ(allclose(t4_hpu.to(torch::kCPU), t4_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t4_hpu.dtype() == t4_cpu.dtype(), true);

  EXPECT_EQ(allclose(t5_hpu.to(torch::kCPU), t5_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t5_hpu.dtype() == t5_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, IndirectInputCast) {
  auto t1_cpu = torch::tensor({22.0}, torch::kInt8);
  auto t2_cpu = torch::tensor({22.0}, torch::kBFloat16);
  auto t3_cpu = torch::mul(t1_cpu, t2_cpu);

  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  auto t3_hpu = torch::mul(t1_hpu, t2_hpu);

  EXPECT_EQ(allclose(t3_hpu.to(torch::kCPU), t3_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t3_hpu.dtype() == t3_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, BoolInt8CastInputCast) {
  auto t1_cpu = torch::tensor({1}, torch::kBool);
  auto t2_cpu = torch::tensor({1}, torch::kInt8);
  auto t3_cpu = torch::mul(t1_cpu, t2_cpu);

  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  auto t3_hpu = torch::mul(t1_hpu, t2_hpu);

  EXPECT_EQ(allclose(t3_hpu.to(torch::kCPU), t3_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t3_hpu.dtype() == t3_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, ClampMin) {
  auto scalar = 2.1f;
  auto t1_cpu = torch::tensor({1, 2, 3, 4}, torch::kInt32);
  auto t2_cpu = torch::tensor({scalar}, torch::kFloat);
  auto t3_cpu = torch::clamp_min(t1_cpu, scalar);
  auto t4_cpu = torch::clamp_min(t1_cpu, t2_cpu);

  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  auto t3_hpu = torch::clamp_min(t1_hpu, scalar);
  auto t4_hpu = torch::clamp_min(t1_hpu, t2_hpu);

  EXPECT_EQ(allclose(t3_hpu.to(torch::kCPU), t3_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t3_hpu.dtype() == t3_cpu.dtype(), true);

  EXPECT_EQ(allclose(t4_hpu.to(torch::kCPU), t4_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t4_hpu.dtype() == t4_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, ClampMax) {
  auto scalar = 2.1f;
  auto t1_cpu = torch::tensor({1, 2, 3, 4}, torch::kInt32);
  auto t2_cpu = torch::tensor({scalar}, torch::kFloat);
  auto t3_cpu = torch::clamp_max(t1_cpu, scalar);
  auto t4_cpu = torch::clamp_max(t1_cpu, t2_cpu);

  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  auto t3_hpu = torch::clamp_max(t1_hpu, scalar);
  auto t4_hpu = torch::clamp_max(t1_hpu, t2_hpu);

  EXPECT_EQ(allclose(t3_hpu.to(torch::kCPU), t3_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t3_hpu.dtype() == t3_cpu.dtype(), true);

  EXPECT_EQ(allclose(t4_hpu.to(torch::kCPU), t4_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t4_hpu.dtype() == t4_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, ClampMaxMin) {
  auto min = -0.5f;
  auto max = 0.5f;
  auto t1_cpu = torch::tensor({1, 2, 3, 4}, torch::kInt32);
  auto min_cpu = torch::tensor({min}, torch::kFloat);
  auto max_cpu = torch::tensor({max}, torch::kFloat);
  auto t3_cpu = torch::clamp(t1_cpu, min_cpu, max_cpu);

  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto min_hpu = min_cpu.to(torch::kHPU);
  auto max_hpu = max_cpu.to(torch::kHPU);
  auto t3_hpu = torch::clamp(t1_hpu, min_hpu, max_hpu);

  EXPECT_EQ(allclose(t3_hpu.to(torch::kCPU), t3_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t3_hpu.dtype() == t3_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, ClampMaxMinScalar) {
  auto min = -0.5f;
  auto max = 0.5f;
  auto t1_cpu = torch::tensor({1, 2, 3, 4}, torch::kInt32);
  auto t3_cpu = torch::clamp(t1_cpu, min, max);

  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t3_hpu = torch::clamp(t1_hpu, min, max);

  EXPECT_EQ(allclose(t3_hpu.to(torch::kCPU), t3_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t3_hpu.dtype() == t3_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, InplaceClampMin) {
  auto scalar = 2;
  auto t1_cpu = torch::tensor({1, 2, 3, 4}, torch::kFloat);
  auto t2_cpu = torch::tensor({1, 2, 3, 4}, torch::kFloat);
  auto t3_cpu = torch::tensor({scalar}, torch::kInt32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  auto t3_hpu = t3_cpu.to(torch::kHPU);

  t1_cpu.clamp_min_(scalar);
  t2_cpu.clamp_min_(t3_cpu);
  t1_hpu.clamp_min_(scalar);
  t2_hpu.clamp_min_(t3_hpu);

  EXPECT_EQ(allclose(t1_hpu.to(torch::kCPU), t1_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t1_hpu.dtype() == t1_cpu.dtype(), true);

  EXPECT_EQ(allclose(t2_hpu.to(torch::kCPU), t2_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t2_hpu.dtype() == t2_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, InplaceClampMax) {
  auto scalar = 2;
  auto t1_cpu = torch::tensor({1, 2, 3, 4}, torch::kFloat);
  auto t2_cpu = torch::tensor({1, 2, 3, 4}, torch::kFloat);
  auto t3_cpu = torch::tensor({scalar}, torch::kInt32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  auto t3_hpu = t3_cpu.to(torch::kHPU);

  t1_cpu.clamp_max_(scalar);
  t2_cpu.clamp_max_(t3_cpu);
  t1_hpu.clamp_max_(scalar);
  t2_hpu.clamp_max_(t3_hpu);

  EXPECT_EQ(allclose(t1_hpu.to(torch::kCPU), t1_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t1_hpu.dtype() == t1_cpu.dtype(), true);

  EXPECT_EQ(allclose(t2_hpu.to(torch::kCPU), t2_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t2_hpu.dtype() == t2_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, OutClampMin) {
  auto scalar = 2;
  auto t1_cpu = torch::tensor({1, 2, 3, 4}, torch::kFloat);
  auto t2_cpu = torch::tensor({1, 2, 3, 4}, torch::kFloat);
  auto t3_cpu = torch::tensor({scalar}, torch::kInt32);
  auto t4_cpu = torch::tensor({}, torch::kFloat);
  auto t5_cpu = torch::tensor({}, torch::kFloat);

  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t3_hpu = t3_cpu.to(torch::kHPU);
  auto t4_hpu = t4_cpu.to(torch::kHPU);
  auto t5_hpu = t5_cpu.to(torch::kHPU);

  torch::clamp_min_out(t4_cpu, t1_cpu, scalar);
  torch::clamp_min_out(t5_cpu, t1_cpu, t3_cpu);

  torch::clamp_min_out(t4_hpu, t1_hpu, scalar);
  torch::clamp_min_out(t5_hpu, t1_hpu, t3_hpu);

  EXPECT_EQ(allclose(t4_hpu.to(torch::kCPU), t4_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t4_hpu.dtype() == t4_cpu.dtype(), true);

  EXPECT_EQ(allclose(t5_hpu.to(torch::kCPU), t5_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t5_hpu.dtype() == t5_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, OutClampMax) {
  auto scalar = 2;
  auto t1_cpu = torch::tensor({1, 2, 3, 4}, torch::kFloat);
  auto t3_cpu = torch::tensor({scalar}, torch::kInt32);
  auto t4_cpu = torch::tensor({}, torch::kFloat);
  auto t5_cpu = torch::tensor({}, torch::kFloat);

  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t3_hpu = t3_cpu.to(torch::kHPU);
  auto t4_hpu = t4_cpu.to(torch::kHPU);
  auto t5_hpu = t5_cpu.to(torch::kHPU);

  torch::clamp_max_out(t4_cpu, t1_cpu, scalar);
  torch::clamp_max_out(t5_cpu, t1_cpu, t3_cpu);

  torch::clamp_max_out(t4_hpu, t1_hpu, scalar);
  torch::clamp_max_out(t5_hpu, t1_hpu, t3_hpu);

  EXPECT_EQ(allclose(t4_hpu.to(torch::kCPU), t4_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t4_hpu.dtype() == t4_cpu.dtype(), true);

  EXPECT_EQ(allclose(t5_hpu.to(torch::kCPU), t5_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t5_hpu.dtype() == t5_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, NegativeOutputCast) {
  auto scalar = 2;
  auto t1 = torch::tensor({1, 2, 3, 4}, torch::kFloat).to(torch::kHPU);
  auto t2 = torch::tensor({0}, torch::kInt32).to(torch::kHPU);

  EXPECT_THROW(torch::add_out(t2, t1, scalar), c10::Error);
  EXPECT_THROW(t2.add_(t1), c10::Error);

  EXPECT_THROW(torch::mul_out(t2, t1, scalar), c10::Error);
  EXPECT_THROW(t2.mul_(t1), c10::Error);

  EXPECT_THROW(torch::sub_out(t2, t1, scalar), c10::Error);
  EXPECT_THROW(t2.sub_(t1), c10::Error);

  EXPECT_THROW(torch::clamp_min_out(t2, t1, scalar), c10::Error);
  EXPECT_THROW(t2.clamp_min_(t1), c10::Error);

  EXPECT_THROW(torch::clamp_max_out(t2, t1, scalar), c10::Error);
  EXPECT_THROW(t2.clamp_max_(t1), c10::Error);

  EXPECT_THROW(
      torch::mean_out(t2, t1, {0}, false, torch::kBFloat16), c10::Error);
}

TEST_F(TypePromotionTests, OutGreaterThan) {
  auto t1_cpu = torch::tensor({22.0}, torch::kInt32);
  auto t2_cpu = torch::tensor({22.0}, torch::kBFloat16);
  auto t3_cpu = torch::tensor({0}, torch::kBool);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  auto t3_hpu = t3_cpu.to(torch::kHPU);
  torch::gt_out(t3_cpu, t1_cpu, t2_cpu);
  torch::gt_out(t3_hpu, t1_hpu, t2_hpu);

  EXPECT_EQ(allclose(t3_hpu.to(torch::kCPU), t3_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t3_hpu.dtype() == t3_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, OutMean) {
  auto t1_cpu = torch::tensor({1, 2}, torch::kFloat);
  auto t2_cpu = torch::tensor({}, torch::kBFloat16);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto t2_hpu = t2_cpu.to(torch::kHPU);
  torch::mean_out(t2_cpu, t1_cpu, {0}, false, torch::kBFloat16);
  torch::mean_out(t2_hpu, t1_hpu, {0}, false, torch::kBFloat16);

  EXPECT_EQ(allclose(t2_hpu.to(torch::kCPU), t2_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t2_hpu.dtype() == t2_cpu.dtype(), true);
}

TEST_F(TypePromotionTests, CumprodInt32) {
  auto t1_cpu = torch::tensor({3}, torch::kInt32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto cpu_out = torch::cumprod(t1_cpu, 0);
  auto hpu_out = torch::cumprod(t1_hpu, 0);

  auto htc_out = hpu_out.to(torch::kCPU);
  EXPECT_TRUE(htc_out.dtype() == torch::kLong);
  EXPECT_EQ(allclose(htc_out, cpu_out, 0.001, 0.001), true);
}

TEST_F(TypePromotionTests, Cumprod0dInt32) {
  auto t1_cpu = torch::tensor(3, torch::kInt32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto cpu_out = torch::cumprod(t1_cpu, 0);
  auto hpu_out = torch::cumprod(t1_hpu, 0);

  auto htc_out = hpu_out.to(torch::kCPU);
  EXPECT_TRUE(htc_out.dtype() == torch::kLong);
  EXPECT_EQ(allclose(htc_out, cpu_out, 0.001, 0.001), true);
}

TEST_F(TypePromotionTests, CumprodInt32dtyLong) {
  auto t1_cpu = torch::tensor({3}, torch::kInt32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto cpu_out = torch::cumsum(t1_cpu, 0, torch::kLong);
  auto hpu_out = torch::cumsum(t1_hpu, 0, torch::kLong);

  auto htc_out = hpu_out.to(torch::kCPU);
  EXPECT_TRUE(htc_out.dtype() == torch::kLong);
  EXPECT_EQ(allclose(htc_out, cpu_out, 0.001, 0.001), true);
}

TEST_F(TypePromotionTests, CumprodLong) {
  auto t1_cpu = torch::tensor({3}, torch::kLong);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto cpu_out = torch::cumsum(t1_cpu, 0);
  auto hpu_out = torch::cumsum(t1_hpu, 0);

  auto htc_out = hpu_out.to(torch::kCPU);
  EXPECT_TRUE(htc_out.dtype() == torch::kLong);
  EXPECT_EQ(allclose(htc_out, cpu_out, 0.001, 0.001), true);
}

TEST_F(TypePromotionTests, CumsumInt32) {
  auto t1_cpu = torch::tensor({3}, torch::kInt32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto cpu_out = torch::cumsum(t1_cpu, 0);
  auto hpu_out = torch::cumsum(t1_hpu, 0);

  auto htc_out = hpu_out.to(torch::kCPU);
  EXPECT_TRUE(htc_out.dtype() == torch::kLong);
  EXPECT_EQ(allclose(htc_out, cpu_out, 0.001, 0.001), true);
}

TEST_F(TypePromotionTests, Cumsum0dInt32) {
  auto t1_cpu = torch::tensor(3, torch::kInt32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto cpu_out = torch::cumsum(t1_cpu, 0);
  auto hpu_out = torch::cumsum(t1_hpu, 0);

  auto htc_out = hpu_out.to(torch::kCPU);
  EXPECT_TRUE(htc_out.dtype() == torch::kLong);
  EXPECT_EQ(allclose(htc_out, cpu_out, 0.001, 0.001), true);
}

TEST_F(TypePromotionTests, CumsumInt32dtyLong) {
  auto t1_cpu = torch::tensor({3}, torch::kInt32);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto cpu_out = torch::cumsum(t1_cpu, 0, torch::kLong);
  auto hpu_out = torch::cumsum(t1_hpu, 0, torch::kLong);

  auto htc_out = hpu_out.to(torch::kCPU);
  EXPECT_TRUE(htc_out.dtype() == torch::kLong);
  EXPECT_EQ(allclose(htc_out, cpu_out, 0.001, 0.001), true);
}

TEST_F(TypePromotionTests, CumsumLong) {
  auto t1_cpu = torch::tensor({3}, torch::kLong);
  auto t1_hpu = t1_cpu.to(torch::kHPU);
  auto cpu_out = torch::cumsum(t1_cpu, 0);
  auto hpu_out = torch::cumsum(t1_hpu, 0);

  auto htc_out = hpu_out.to(torch::kCPU);
  EXPECT_TRUE(htc_out.dtype() == torch::kLong);
  EXPECT_EQ(allclose(htc_out, cpu_out, 0.001, 0.001), true);
}

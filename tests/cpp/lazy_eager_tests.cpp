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
#include "habana_lazy_test_infra.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include "habana_kernels/lazy_kernels_declarations.h"

#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"

#include "hpu_ops/util.h"

using namespace habana_lazy;

// In this class both the pass fallback and compilation fallback are disabled
class DISABLED_LazyEagerTest : public HpuOpTestUtil {
  void SetUp() override {
    SetLazyMode(2);

    DisableRecipeCache();
    DisableAccParMode();

    SetSeed();

    DisableCpuFallback();

    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
  }

  void TearDown() override {
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();

    RestoreRecipeCache();
    RestoreAccParMode();

    RestoreMode();
  }
};

TEST_F(DISABLED_LazyEagerTest, optimized_lazy_eager_log_sigmoid_fwd_out_1) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPUDeviceContext::get_device();
  if (device.type() == synDeviceGaudi2) {
    auto out = torch::empty(0);
    auto hout = torch::empty(0, c10::kHPU);
    auto buffer = torch::empty(0);
    auto hbuffer = torch::empty(0, c10::kHPU);
    GenerateInputs(1);
    torch::log_sigmoid_forward_outf(
        GetHpuInput(0), hout, hbuffer); // for cache miss
    torch::log_sigmoid_forward_outf(GetCpuInput(0), out, buffer);
    Compare(out, hout);
  }
}

TEST_F(DISABLED_LazyEagerTest, optimized_lazy_eager_log_sigmoid_fwd_out_2) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPUDeviceContext::get_device();
  if (device.type() == synDeviceGaudi2) {
    const int iterations = 10;
    auto out = torch::empty(0);
    auto hout = torch::empty(0, c10::kHPU);
    auto buffer = torch::empty(0);
    auto hbuffer = torch::empty(0, c10::kHPU);
    GenerateInputs(1);
    torch::log_sigmoid_forward_outf(
        GetHpuInput(0), hout, hbuffer); // for cache miss
    torch::log_sigmoid_forward_outf(GetCpuInput(0), out, buffer);
    Compare(out, hout);
    for (int i = 0; i < iterations; i++) {
      torch::log_sigmoid_forward_outf(GetHpuInput(0), hout, hbuffer);
      torch::log_sigmoid_forward_outf(GetCpuInput(0), out, buffer);
      Compare(out, hout);
    }
  }
}

TEST_F(DISABLED_LazyEagerTest, optimized_lazy_eager_mul_inplace_1) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPUDeviceContext::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({2, 3});
    torch::Tensor B = torch::randn({2, 3});
    torch::Tensor C = torch::randn({2, 3});
    auto hA = A.to(torch::kHPU);
    auto hB = B.to(torch::kHPU);
    auto hC = C.to(torch::kHPU);
    A = A.mul_(B);
    auto exp = torch::add(A, C);
    hA = hA.mul_(hB);
    auto result = torch::add(hA, hC);
    torch::Tensor out = result.to(torch::kCPU);
    EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
  }
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(DISABLED_LazyEagerTest, optimized_lazy_eager_mul_inplace_2) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPUDeviceContext::get_device();
  if (device.type() == synDeviceGaudi2) {
    SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, 2, 1);
    SET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE, 0, 1);
    torch::Tensor A = torch::randn({2, 3});
    torch::Tensor B = torch::randn({2, 3});
    torch::Tensor C = torch::randn({2, 3});
    auto hA = A.to(torch::kHPU);
    auto hB = B.to(torch::kHPU);
    auto hC = C.to(torch::kHPU);
    A = A.mul_(B);
    hA = hA.mul_(hB);
    torch::Tensor out = hA.to(torch::kCPU);
    EXPECT_EQ(allclose(out, A, 0.001, 0.001), true);
    long long total_time = 0;
    const int iterations = 10;
    for (int i = 0; i < iterations; i++) {
      A = A.mul_(B);
      auto start = std::chrono::high_resolution_clock::now();
      hA = hA.mul_(hB);
      auto elapsed = std::chrono::high_resolution_clock::now() - start;
      torch::Tensor out = hA.to(torch::kCPU);
      total_time +=
          std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
              .count();
      EXPECT_EQ(allclose(out, hA, 0.001, 0.001), true);
    }
  }
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(DISABLED_LazyEagerTest, optimized_lazy_copy_inplace_1) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPUDeviceContext::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor tensor_A = torch::randn({1, 3, 3, 3});
    torch::Tensor tensor_B = torch::randn({1, 3, 3, 3});
    torch::Tensor tensor_C = torch::randn({1, 3, 3, 3});
    auto tensor_A_bf16 = tensor_A.to(torch::kBFloat16);
    auto tensor_B_bf16 = tensor_B.to(torch::kBFloat16);
    auto tensor_C_bf16 = tensor_C.to(torch::kBFloat16);

    torch::Tensor tHabana_A = tensor_A_bf16.to(torch::kHPU);
    torch::Tensor tHabana_B = tensor_B_bf16.to(torch::kHPU);
    torch::Tensor tHabana_C = tensor_C_bf16.to(torch::kHPU);

    for (int i = 0; i < 3; i++) {
      tensor_B = tensor_B.add_(tensor_A);
      tensor_B = relu_(tensor_B);
      tensor_C.copy_(tensor_B);
      tHabana_B = tHabana_B.add_(tHabana_A);
      tHabana_B = relu_(tHabana_B);
      tHabana_C.copy_(tHabana_B);
      torch::Tensor out = tHabana_C.to(torch::kFloat).to(torch::kCPU);
      EXPECT_EQ(allclose(out, tensor_C, 0.01, 0.01), true);
    }
  }
}

TEST_F(DISABLED_LazyEagerTest, optimized_lazy_eager_add_f32_with_scalar) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPUDeviceContext::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A =
        torch::randn({2, 3}, torch::dtype(torch::kFloat32).requires_grad(false))
            .to(torch::kLong);
    auto hA = A.to(torch::kHPU);
    A = A.add_(1);
    auto exp = torch::add(A, 3);

    hA = hA.add_(1);
    auto result = torch::add(hA, 3);
    torch::Tensor out = result.to(torch::kCPU);
    EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  }
}

TEST_F(DISABLED_LazyEagerTest, optimized_lazy_eager_add_i32_with_scalar) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPUDeviceContext::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A =
        torch::randint(
            1, 9, {3, 3}, torch::dtype(torch::kInt32).requires_grad(false))
            .to(torch::kLong);
    auto hA = A.to(torch::kHPU);
    A = A.add_(1);
    auto exp = torch::add(A, 3, 2);

    hA = hA.add_(1);
    auto result = torch::add(hA, 3, 2);
    torch::Tensor out = result.to(torch::kCPU);
    EXPECT_EQ(allclose(out, exp, 0, 0), true);
  }
}

TEST_F(DISABLED_LazyEagerTest, optimized_lazy_eager_div_f32_with_scalar) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPUDeviceContext::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn(
        {3, 3}, torch::dtype(torch::kFloat32).requires_grad(false));
    auto hA = A.to(torch::kHPU);
    auto B = torch::empty(0);
    auto hB = torch::empty(0, torch::kHPU);

    A = A.div_(5);
    torch::div_outf(A, 2, B);
    auto exp = torch::div(B, 3);

    hA = hA.div_(5);
    torch::div_outf(hA, 2, hB);
    auto result = torch::div(hB, 3);
    torch::Tensor out = result.to(torch::kCPU);

    EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  }
}

TEST_F(DISABLED_LazyEagerTest, optimized_lazy_eager_div_mode_bf16_with_scalar) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPUDeviceContext::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn(
        {3, 3}, torch::dtype(torch::kBFloat16).requires_grad(false));
    auto hA = A.to(torch::kHPU);

    A = A.div_(5, "floor");
    auto exp = torch::div(A, 3, "floor");

    hA = hA.div_(5, "floor");
    auto result = torch::div(hA, 3, "floor");
    torch::Tensor out = result.to(torch::kCPU);
    EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  }
}

TEST_F(DISABLED_LazyEagerTest, optimized_lazy_eager_div_mode_i32_with_scalar) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPUDeviceContext::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randint(
        1, 100, {10, 10}, torch::dtype(torch::kInt32).requires_grad(false));
    auto hA = A.to(torch::kHPU);

    A = A.div_(2, "trunc");
    auto exp = torch::div(A, 4, "trunc");

    hA = hA.div_(2, "trunc");
    auto result = torch::div(hA, 4, "trunc");
    torch::Tensor out = result.to(torch::kCPU);
    EXPECT_EQ(allclose(out, exp, 0, 0), true);
  }
}

TEST_F(DISABLED_LazyEagerTest, optimized_lazy_eager_clamp_with_scalar) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPUDeviceContext::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randint(
        -50, 50, {10, 10}, torch::dtype(torch::kInt32).requires_grad(false));
    auto hA = A.to(torch::kHPU);

    torch::Scalar s1 = -10;
    torch::Scalar s2 = 10;
    auto exp = torch::clamp(A, s1, s2);
    auto result = torch::clamp(hA, s1, s2);

    torch::Tensor out = result.to(torch::kCPU);
    EXPECT_EQ(allclose(out, exp, 0, 0), true);
  }
}

TEST_F(DISABLED_LazyEagerTest, optimized_lazy_eager_cmp_with_scalar) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPUDeviceContext::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randint(
        0, 2, {10}, torch::dtype(torch::kInt32).requires_grad(false));
    auto hA = A.to(torch::kHPU);

    torch::Scalar scalar = 1;
    auto exp = torch::eq(A, scalar);
    auto result = torch::eq(hA, scalar);

    torch::Tensor out = result.to(torch::kCPU);
    EXPECT_EQ(allclose(out, exp, 0, 0), true);
  }
}

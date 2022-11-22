/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_lazy_test_infra.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include "habana_kernels/lazy_kernels_declarations.h"

#include "pytorch_helpers/habana_device/HPUGuardImpl.h"
#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/habana_helpers/tensor_utils.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

#include "hpu_ops/util.h"

using namespace habana_lazy;

// In this class both the pass fallback and compilation fallback are disabled
class LazyEagerTest : public HpuOpTestUtil {};

TEST_F(LazyEagerTest, optimized_lazy_eager_log_sigmoid_fwd_out_1) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, 2, 1);
    SET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE, 0, 1);
    SET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_SYN_API, true, 1);
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

TEST_F(LazyEagerTest, optimized_lazy_eager_log_sigmoid_fwd_out_2) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, 2, 1);
    SET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE, 0, 1);
    SET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_SYN_API, true, 1);
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

TEST_F(LazyEagerTest, optimized_lazy_eager_mul_inplace_1) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, 2, 1);
    SET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE, 0, 1);
    SET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_SYN_API, true, 1);
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
    UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
  }
}

TEST_F(LazyEagerTest, optimized_lazy_eager_mul_inplace_2) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, 2, 1);
    SET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE, 0, 1);
    SET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_SYN_API, true, 1);
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
    UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
  }
}

TEST_F(LazyEagerTest, optimized_lazy_copy_inplace_1) {
  SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, 2, 1);
  SET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE, 0, 1);
  SET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_SYN_API, true, 1);
  SET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_PAR_MODE, false, 1);
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
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
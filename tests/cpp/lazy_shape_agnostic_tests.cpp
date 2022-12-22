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

using namespace habana_lazy;

// In this class both the pass fallback and compilation fallback are disabled
class LazyShapeAgnosticTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    SetLazyMode(2);

    DisableRecipeCache();
    EnableEagerGC();
    EnableShapeAgnostic();
    DisableAccParMode();

    SetSeed();

    DisableCpuFallback();

    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();

    habana_lazy::StageSubmission::getInstance().resetCurrentAccumulatedOps();
  }

  void TearDown() override {
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();

    RestoreRecipeCache();
    RestoreEagerGC();
    RestoreShapeAgnostic();
    RestoreAccParMode();

    RestoreMode();
  }
};

TEST_F(LazyShapeAgnosticTest, LazyDoATest1) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({5, 5}, torch::requires_grad(false));
    torch::Tensor B = torch::randn({5, 5}, torch::requires_grad(false));
    torch::Tensor C = torch::randn({5, 5}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hC = C.to(torch::kHPU);
    torch::Tensor I = torch::add(hA, hB, 2.3);
    torch::Tensor out = torch::add(hC, I, 2.3);

    torch::Tensor A_2 = torch::randn({10, 10}, torch::requires_grad(false));
    torch::Tensor B_2 = torch::randn({10, 10}, torch::requires_grad(false));
    torch::Tensor C_2 = torch::randn({10, 10}, torch::requires_grad(false));
    torch::Tensor hA_2 = A_2.to(torch::kHPU);
    torch::Tensor hB_2 = B_2.to(torch::kHPU);
    torch::Tensor hC_2 = C_2.to(torch::kHPU);
    torch::Tensor I_2 = torch::add(hA_2, hB_2, 2.3);
    torch::Tensor out_2 = torch::add(hC_2, I_2, 2.3);

    torch::Tensor I_cpu = torch::add(A, B, 2.3);
    torch::Tensor out_cpu = torch::add(C, I_cpu, 2.3);
    torch::Tensor out_h = out.to(torch::kCPU);
    torch::Tensor I_cpu_2 = torch::add(A_2, B_2, 2.3);
    torch::Tensor out_cpu_2 = torch::add(C_2, I_cpu_2, 2.3);
    torch::Tensor out_h_2 = out_2.to(torch::kCPU);

    EXPECT_EQ(allclose(out_h, out_cpu, 0.001, 0.001), true);
    EXPECT_EQ(allclose(out_h_2, out_cpu_2, 0.001, 0.001), true);
  }
}

TEST_F(LazyShapeAgnosticTest, LazyDoATest2) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({5, 5}, torch::requires_grad(false));
    torch::Tensor B = torch::randn({5, 5}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor out = torch::add(hA, hB);

    torch::Tensor A_2 = torch::randn({3, 3}, torch::requires_grad(false));
    torch::Tensor B_2 = torch::randn({3, 3}, torch::requires_grad(false));
    torch::Tensor hA_2 = A_2.to(torch::kHPU);
    torch::Tensor hB_2 = B_2.to(torch::kHPU);
    torch::Tensor out_2 = torch::add(hA_2, hB_2);

    torch::Tensor A_3 = torch::randn({6, 6}, torch::requires_grad(false));
    torch::Tensor B_3 = torch::randn({6, 6}, torch::requires_grad(false));
    torch::Tensor hA_3 = A_3.to(torch::kHPU);
    torch::Tensor hB_3 = B_3.to(torch::kHPU);
    torch::Tensor out_3 = torch::add(hA_3, hB_3);

    torch::Tensor out_cpu = torch::add(A, B);
    torch::Tensor out_h = out.to(torch::kCPU);
    torch::Tensor out_cpu_2 = torch::add(A_2, B_2);
    torch::Tensor out_h_2 = out_2.to(torch::kCPU);
    torch::Tensor out_cpu_3 = torch::add(A_3, B_3);
    torch::Tensor out_h_3 = out_3.to(torch::kCPU);

    EXPECT_EQ(allclose(out_h, out_cpu, 0.001, 0.001), true);
    EXPECT_EQ(allclose(out_h_2, out_cpu_2, 0.001, 0.001), true);
    EXPECT_EQ(allclose(out_h_3, out_cpu_3, 0.001, 0.001), true);
  }
}

TEST_F(LazyShapeAgnosticTest, LazyDoATest3) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({5, 10}, torch::requires_grad(false));
    torch::Tensor B = torch::randn({5, 10}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor out = torch::add(hA, hB);

    torch::Tensor A_2 = torch::randn({3, 3}, torch::requires_grad(false));
    torch::Tensor B_2 = torch::randn({3, 3}, torch::requires_grad(false));
    torch::Tensor hA_2 = A_2.to(torch::kHPU);
    torch::Tensor hB_2 = B_2.to(torch::kHPU);
    torch::Tensor out_2 = torch::add(hA_2, hB_2);

    torch::Tensor out_cpu = torch::add(A, B);
    torch::Tensor out_h = out.to(torch::kCPU);
    torch::Tensor out_cpu_2 = torch::add(A_2, B_2);
    torch::Tensor out_h_2 = out_2.to(torch::kCPU);

    EXPECT_EQ(allclose(out_h, out_cpu, 0.001, 0.001), true);
    EXPECT_EQ(allclose(out_h_2, out_cpu_2, 0.001, 0.001), true);
  }
}

TEST_F(LazyShapeAgnosticTest, ConvReluTest1) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    auto input_tensor =
        torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 3, 3, 3}); // nchw
    torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
    auto input_tensor_2 =
        torch::arange(64, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 4, 4, 4}); // nchw
    torch::Tensor tHabanaX_2 = input_tensor_2.to(torch::kHPU);

    auto weight_tensor =
        torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({3, 3, 3, 1}); // hwck
    auto weight_tensor_2 =
        torch::arange(64, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({4, 4, 4, 1}); // hwck

    torch::Tensor tHabanaW = weight_tensor.to(torch::kHPU);
    torch::Tensor tHabanaW_2 = weight_tensor_2.to(torch::kHPU);
    if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) &&
        !habana_lazy::exec::OptPassCfg::GetInstance()
             ->IsEnabledWeightPermutePass()) {
      auto wt_hwck = weight_tensor.permute({2, 3, 1, 0}).contiguous();
      tHabanaW = wt_hwck.to(torch::kHPU);
      auto wt_hwck_2 = weight_tensor_2.permute({2, 3, 1, 0}).contiguous();
      tHabanaW_2 = wt_hwck_2.to(torch::kHPU);
    }

    torch::Tensor outConv =
        torch::conv2d(tHabanaX, tHabanaW, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor outConv_2 = torch::conv2d(
        tHabanaX_2, tHabanaW_2, {}, {1}, at::IntArrayRef{0}, {1}, 1);

    torch::Tensor outhpu = torch::relu(outConv);
    torch::Tensor outhpu_2 = torch::relu(outConv_2);

    torch::Tensor out = outhpu.to(torch::kCPU);
    torch::Tensor out_2 = outhpu_2.to(torch::kCPU);
    torch::Tensor out_conv = outConv.to(torch::kCPU);
    torch::Tensor out_conv_2 = outConv_2.to(torch::kCPU);

    torch::Tensor outConv1 = torch::conv2d(
        input_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor outcpu = torch::relu(outConv1);

    torch::Tensor outConv2 = torch::conv2d(
        input_tensor_2, weight_tensor_2, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor outcpu_2 = torch::relu(outConv2);

    EXPECT_EQ(allclose(out_conv, outConv1, 0.01, 0.01), true);
    EXPECT_EQ(allclose(out_conv_2, outConv2, 0.01, 0.01), true);
    EXPECT_EQ(allclose(out, outcpu, 0.01, 0.01), true);
    EXPECT_EQ(allclose(out_2, outcpu_2, 0.01, 0.01), true);
  }
}

TEST_F(LazyShapeAgnosticTest, ConvReluTest2) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    size_t test_count = 10;
    for (size_t count = 0; count < test_count; count++) {
      auto input_tensor =
          torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
              .reshape({1, 3, 3, 3}); // nchw
      torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
      auto input_tensor_2 =
          torch::arange(64000, torch::dtype(torch::kFloat).requires_grad(false))
              .reshape({1, 40, 40, 40}); // nchw
      torch::Tensor tHabanaX_2 = input_tensor_2.to(torch::kHPU);

      auto weight_tensor =
          torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
              .reshape({3, 3, 3, 1}); // hwck
      auto weight_tensor_2 =
          torch::arange(64000, torch::dtype(torch::kFloat).requires_grad(false))
              .reshape({40, 40, 40, 1}); // hwck

      torch::Tensor tHabanaW = weight_tensor.to(torch::kHPU);
      torch::Tensor tHabanaW_2 = weight_tensor_2.to(torch::kHPU);
      if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) &&
          !habana_lazy::exec::OptPassCfg::GetInstance()
               ->IsEnabledWeightPermutePass()) {
        auto wt_hwck = weight_tensor.permute({2, 3, 1, 0}).contiguous();
        tHabanaW = wt_hwck.to(torch::kHPU);
        auto wt_hwck_2 = weight_tensor_2.permute({2, 3, 1, 0}).contiguous();
        tHabanaW_2 = wt_hwck_2.to(torch::kHPU);
      }

      torch::Tensor outConv = torch::conv2d(
          tHabanaX, tHabanaW, {}, {1}, at::IntArrayRef{0}, {1}, 1);
      torch::Tensor outConv_2 = torch::conv2d(
          tHabanaX_2, tHabanaW_2, {}, {1}, at::IntArrayRef{0}, {1}, 1);

      torch::Tensor outhpu = torch::relu(outConv);
      torch::Tensor outhpu_2 = torch::relu(outConv_2);

      torch::Tensor out = outhpu.to(torch::kCPU);
      torch::Tensor out_2 = outhpu_2.to(torch::kCPU);

      torch::Tensor outConv1 = torch::conv2d(
          input_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
      torch::Tensor outcpu = torch::relu(outConv1);

      torch::Tensor outConv2 = torch::conv2d(
          input_tensor_2, weight_tensor_2, {}, {1}, at::IntArrayRef{0}, {1}, 1);
      torch::Tensor outcpu_2 = torch::relu(outConv2);

      EXPECT_EQ(allclose(out, outcpu, 0.01, 0.01), true);
      EXPECT_EQ(allclose(out_2, outcpu_2, 0.01, 0.01), true);
    }
  }
}

TEST_F(LazyShapeAgnosticTest, MulInplaceTest) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
      SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
    }
    // Inplace op as output node is not supported yet.
    torch::Tensor A = torch::randn({2, 3});
    torch::Tensor B = torch::randn({2, 3});
    torch::Tensor C = torch::randn({2, 3});
    auto hA = A.to(torch::kHPU);
    auto hB = B.to(torch::kHPU);
    auto hC = C.to(torch::kHPU);
    torch::Tensor A_2 = torch::randn({40, 60});
    torch::Tensor B_2 = torch::randn({40, 60});
    torch::Tensor C_2 = torch::randn({40, 60});
    auto hA_2 = A_2.to(torch::kHPU);
    auto hB_2 = B_2.to(torch::kHPU);
    auto hC_2 = C_2.to(torch::kHPU);

    A = A.mul_(B);
    auto exp = torch::add(A, C);
    A_2 = A_2.mul_(B_2);
    auto exp_2 = torch::add(A_2, C_2);

    hA = hA.mul_(hB);
    auto result = torch::add(hA, hC);
    torch::Tensor out = result.to(torch::kCPU);
    hA_2 = hA_2.mul_(hB_2);
    auto result_2 = torch::add(hA_2, hC_2);
    torch::Tensor out_2 = result_2.to(torch::kCPU);

    EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
    EXPECT_EQ(allclose(out_2, exp_2, 0.001, 0.001), true);
    UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
  }
}

TEST_F(LazyShapeAgnosticTest, BatchNormForwardExecute) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
      SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
    }
    auto input_tensor = torch::randn(
        {10, 3, 4, 2}, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
    auto input_tensor_2 = torch::randn(
        {20, 5, 3, 3}, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor tHabanaX_2 = input_tensor_2.to(torch::kHPU);

    at::Tensor weight =
        torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor tWeight = weight.to(torch::kHPU);
    at::Tensor bias =
        torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor tBias = bias.to(torch::kHPU);
    auto mean =
        torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor tHabanaMean = mean.to(torch::kHPU);
    auto var = torch::ones(3, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor tHabanaVar = var.to(torch::kHPU);

    at::Tensor weight_2 =
        torch::randn(5, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor tWeight_2 = weight_2.to(torch::kHPU);
    at::Tensor bias_2 =
        torch::randn(5, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor tBias_2 = bias_2.to(torch::kHPU);
    auto mean_2 =
        torch::randn(5, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor tHabanaMean_2 = mean_2.to(torch::kHPU);
    auto var_2 =
        torch::ones(5, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor tHabanaVar_2 = var_2.to(torch::kHPU);

    float mom = 0.1;
    float eps = 1e-5;
    // Training = True
    auto results_cpu = torch::native_batch_norm(
        input_tensor, weight, bias, mean, var, true, mom, eps);
    auto results_cpu_2 = torch::native_batch_norm(
        input_tensor_2, weight_2, bias_2, mean_2, var_2, true, mom, eps);

    at::Tensor result_cpu = std::get<0>(results_cpu);
    auto curr_mean_cpu = std::get<1>(results_cpu);
    at::Tensor result_cpu_2 = std::get<0>(results_cpu_2);
    auto curr_mean_cpu_2 = std::get<1>(results_cpu_2);

    auto results = torch::native_batch_norm(
        tHabanaX, tWeight, tBias, tHabanaMean, tHabanaVar, true, mom, eps);
    auto results_2 = torch::native_batch_norm(
        tHabanaX_2,
        tWeight_2,
        tBias_2,
        tHabanaMean_2,
        tHabanaVar_2,
        true,
        mom,
        eps);

    HbLazyTensor::StepMarker({});
    at::Tensor result_lazy = std::get<0>(results).to(torch::kCPU);
    auto curr_mean_lazy = std::get<1>(results).to(torch::kCPU);
    at::Tensor result_lazy_2 = std::get<0>(results_2).to(torch::kCPU);
    auto curr_mean_lazy_2 = std::get<1>(results_2).to(torch::kCPU);

    EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
    EXPECT_EQ(allclose(curr_mean_lazy.cpu(), curr_mean_cpu, 0.01, 0.01), true);
    EXPECT_EQ(allclose(tHabanaMean.cpu(), mean, 0.01, 0.01), true);
    // Note higher tolerance needed for variance due to TPC kernel accuracy
    // limitation
    EXPECT_EQ(allclose(tHabanaVar.cpu(), var, 0.1, 0.1), true);

    EXPECT_EQ(allclose(result_lazy_2, result_cpu_2, 0.01, 0.01), true);
    EXPECT_EQ(
        allclose(curr_mean_lazy_2.cpu(), curr_mean_cpu_2, 0.01, 0.01), true);
    EXPECT_EQ(allclose(tHabanaMean_2.cpu(), mean_2, 0.01, 0.01), true);
    // Note higher tolerance needed for variance due to TPC kernel accuracy
    // limitation
    EXPECT_EQ(allclose(tHabanaVar_2.cpu(), var_2, 0.1, 0.1), true);
    UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
  }
}

TEST_F(LazyShapeAgnosticTest, DISABLED_LayerNormForwardExecute) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    auto input_tensor =
        torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({10, 1, 3, 4, 4}); // nchw
    torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
    at::Tensor weight =
        torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 3, 4, 4}); // nchw;
    torch::Tensor tWeight = weight.to(torch::kHPU);
    at::Tensor bias =
        torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 3, 4, 4}); // nchw;
    torch::Tensor tBias = bias.to(torch::kHPU);
    auto results =
        torch::native_layer_norm(tHabanaX, {1, 3, 4, 4}, tWeight, tBias, 0.01);

    auto input_tensor_2 =
        torch::arange(1920, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({40, 1, 3, 4, 4}); // nchw
    torch::Tensor tHabanaX_2 = input_tensor_2.to(torch::kHPU);
    at::Tensor weight_2 =
        torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 3, 4, 4}); // nchw;
    torch::Tensor tWeight_2 = weight_2.to(torch::kHPU);
    at::Tensor bias_2 =
        torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 3, 4, 4}); // nchw;
    torch::Tensor tBias_2 = bias_2.to(torch::kHPU);
    auto results_2 = torch::native_layer_norm(
        tHabanaX_2, {1, 3, 4, 4}, tWeight_2, tBias_2, 0.01);

    at::Tensor result_lazy = (std::get<0>(results)).to(torch::kCPU);
    at::Tensor result_lazy_2 = (std::get<0>(results_2)).to(torch::kCPU);
    auto results_cpu = torch::native_layer_norm(
        input_tensor, {1, 3, 4, 4}, weight, bias, 0.01);
    auto results_cpu_2 = torch::native_layer_norm(
        input_tensor_2, {1, 3, 4, 4}, weight_2, bias_2, 0.01);
    at::Tensor result_cpu = std::get<0>(results_cpu);
    at::Tensor result_cpu_2 = std::get<0>(results_cpu_2);
    EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
    EXPECT_EQ(allclose(result_lazy_2, result_cpu_2, 0.01, 0.01), true);
  }
}

TEST_F(LazyShapeAgnosticTest, AvgPoolTest) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    auto input_tensor =
        torch::arange(20, torch::dtype(torch::kFloat).requires_grad(true))
            .reshape({1, 1, 4, 5}); // nchw
    auto input_tensor_2 =
        torch::arange(40, torch::dtype(torch::kFloat).requires_grad(true))
            .reshape({1, 1, 8, 5}); // nchw
    auto cpu_out = torch::avg_pool2d(input_tensor, 3, 1);
    auto cpu_out_2 = torch::avg_pool2d(input_tensor_2, 3, 1);

    // fwd propagation
    torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
    torch::Tensor tHabanaX_2 = input_tensor_2.to(torch::kHPU);
    auto outHabana =
        torch::avg_pool2d(tHabanaX, {3, 3}, {1, 1}, {0, 0}, false, true);
    auto outHabana_2 =
        torch::avg_pool2d(tHabanaX_2, {3, 3}, {1, 1}, {0, 0}, false, true);

    ASSERT_TRUE(torch::allclose(outHabana.to(torch::kCPU), cpu_out));
    ASSERT_TRUE(torch::allclose(outHabana_2.to(torch::kCPU), cpu_out_2));

    // bwd propagation with dummy grad tensor
    auto grad_tensor =
        torch::arange(6, torch::dtype(torch::kFloat).requires_grad(true))
            .reshape({1, 1, 2, 3});
    torch::Tensor tHabanaG = grad_tensor.to(torch::kHPU);
    outHabana.backward({tHabanaG}, false, true);

    auto out_cpu_lazy = outHabana.to(torch::kCPU);

    ASSERT_TRUE(torch::allclose(out_cpu_lazy, cpu_out));
  }
}

TEST_F(LazyShapeAgnosticTest, ConvTranspose2dBwdTest) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    SET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_SHAPE_AGNOSTIC_GRAPH, true, 1);
    if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
      SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

    auto in = torch::randn({64, 4, 28, 28}, torch::requires_grad()); // nchw
    auto hin = in.to(torch::kHPU);
    auto wt = torch::randn({4, 5, 3, 3}, torch::requires_grad()); // ckhw
    auto hwt = wt.to(torch::kHPU);
    if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) &&
        !habana_lazy::exec::OptPassCfg::GetInstance()
             ->IsEnabledWeightPermutePass()) {
      auto wt_hwck = wt.detach().permute({2, 3, 1, 0}).contiguous();
      hwt = wt_hwck.to(torch::kHPU);
    }
    auto bias = torch::randn({5}, torch::requires_grad()); // k
    auto exp = torch::conv_transpose2d(in, wt, {}, 1, 0, 0, 1, 1);

    auto grad_out = torch::ones_like(exp.detach());
    auto hgrad_out = grad_out.detach().to(torch::kHPU);
    exp.backward(grad_out);
    auto grad_in = in.grad();
    auto grad_wt = wt.grad();

    torch::Tensor hgrad_in, hgrad_wt, hgrad_bias;
    std::array<bool, 3> mask{1, 1, 0};
    std::tie(hgrad_in, hgrad_wt, hgrad_bias) = convolution_backward_hpu_lazy(
        hgrad_out, hin, hwt, {1, 1}, {0, 0}, {1, 1}, true, {0, 0}, 1, mask);
    std::tie(hgrad_in, hgrad_wt, hgrad_bias) = convolution_backward_hpu_lazy(
        hgrad_out, hin, hwt, {1, 1}, {0, 0}, {1, 1}, true, {0, 0}, 1, mask);

    // TBD: aten::backward is not handled by lazy mode, therefore this is
    // not working. This code can be restored when that is fixed.
    /*auto result = torch::conv_transpose2d(hin, hwt, {}, 1, 0, 0, 1, 1);
    result.backward(hgrad_out);
    auto hgrad_in = hin.grad();
    auto hgrad_wt = hwt.grad();*/

    // without explicit stepmarker here. DMA for hgrad_in tensor gets messed up
    // most likely due to 2 outputs from backward op. TBD: remove this once
    // issue is debugged and fixed. HbLazyTensor::StepMarker({});

    auto hgrad_wt_cpu = hgrad_wt.to(torch::kCPU);
    auto hgrad_in_cpu = hgrad_in.to(torch::kCPU);
    if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) &&
        !habana_lazy::exec::OptPassCfg::GetInstance()
             ->IsEnabledWeightPermutePass()) {
      EXPECT_EQ(
          allclose(grad_wt, hgrad_wt_cpu.permute({3, 2, 0, 1}), 0.01, 0.01),
          true);
    } else {
      EXPECT_EQ(allclose(grad_wt, hgrad_wt_cpu, 0.01, 0.01), true);
    }
    UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
  }
}

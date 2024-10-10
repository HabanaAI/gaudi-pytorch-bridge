/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"

#include "hpu_ops/util.h"

// In this class both the pass fallback and compilation fallback are disabled
class EagerViewOpsTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    SetEagerMode();

    DisableRecipeCache();
    EnableEagerViewHandling();
    DisableAccParMode();

    SetSeed();

    DisableCpuFallback();
  }

  void TearDown() override {

    RestoreRecipeCache();
    RestoreEagerViewHandling();
    RestoreAccParMode();

    RestoreMode();
  }
};

TEST_F(EagerViewOpsTest, AddOnAsStrided1) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({3, 3});
    auto hA = A.to(torch::kHPU);
    auto B = A.as_strided({2, 2}, {1, 2}, 1);
    auto C = B.add(1.0);

    auto hB = hA.as_strided({2, 2}, {1, 2}, 1);

    // To support only as_strided, either we lower it from copy or do it on cpu
    // EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);

    auto hC = hB.add(1.0);

    EXPECT_EQ(allclose(C, hC.cpu(), 0.001, 0.001), true);
  }
}

TEST_F(EagerViewOpsTest, AddOnAsStrided2) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({3, 3});
    auto hA = A.to(torch::kHPU);
    auto B = A.as_strided({2, 2}, {1, 2}, 1);
    auto C = B.add(1.0);

    auto hB = hA.as_strided({2, 2}, {1, 2}, 1);

    // To support only as_strided, either we lower it from copy or do it on cpu
    // EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);

    auto hC = hB.add(1.0);

    EXPECT_EQ(allclose(C, hC.cpu(), 0.001, 0.001), true);

    auto D = A.as_strided({2, 2}, {1, 2}, 0);
    auto E = D.add(1.0);

    auto hD = hA.as_strided({2, 2}, {1, 2}, 0);
    auto hE = hD.add(1.0);

    EXPECT_EQ(allclose(E, hE.cpu(), 0.001, 0.001), true);
  }
}

TEST_F(EagerViewOpsTest, AddOnAsStrided3) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({3, 3});
    auto hA = A.to(torch::kHPU);
    auto B = A.as_strided({2, 2}, {1, 2}, 1);
    auto C = B.add(1.0);

    auto hB = hA.as_strided({2, 2}, {1, 2}, 1);

    // To support only as_strided, either we lower it from copy or do it on cpu
    // EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);

    auto hC = hB.add(1.0);

    EXPECT_EQ(allclose(C, hC.cpu(), 0.001, 0.001), true);

    auto D = A.as_strided({2, 2}, {1, 2}, 0);
    auto E = D.add(1.0);

    auto hD = hA.as_strided({2, 2}, {1, 2}, 0);
    auto hE = hD.add(1.0);

    EXPECT_EQ(allclose(E, hE.cpu(), 0.001, 0.001), true);

    D = A.as_strided({2, 2}, {1, 2}, 1);
    E = D.add(1.0);

    hD = hA.as_strided({2, 2}, {1, 2}, 1);
    hE = hD.add(1.0);

    EXPECT_EQ(allclose(E, hE.cpu(), 0.001, 0.001), true);
  }
}

TEST_F(EagerViewOpsTest, AddOnView1) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({4});
    auto B = A.view({2, 2});
    auto C = B.add(1.0);
    auto hA = A.to(torch::kHPU);
    auto hB = hA.view({2, 2});
    auto hC = hB.add(1.0);

    EXPECT_EQ(allclose(C, hC.cpu(), 0.001, 0.001), true);
  }
}

TEST_F(EagerViewOpsTest, AddOnView2) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({6});
    auto B = A.view({2, 3});
    auto C = torch::transpose(B, 0, 1);
    auto D = C.add(1.0);
    auto hA = A.to(torch::kHPU);
    auto hB = hA.view({2, 3});
    auto hC = torch::transpose(hB, 0, 1);
    auto hD = hC.add(1.0);

    EXPECT_EQ(allclose(D, hD.cpu(), 0.001, 0.001), true);
  }
}

TEST_F(EagerViewOpsTest, AddOnView3) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({6});
    auto B = A.view({2, 3});
    auto C = torch::transpose(B, 0, 1);
    auto D = C.add(1.0);
    auto hA = A.to(torch::kHPU);
    auto hB = hA.view({2, 3});
    auto hC = torch::transpose(hB, 0, 1);
    auto hD = hC.add(1.0);

    EXPECT_EQ(allclose(D, hD.cpu(), 0.001, 0.001), true);
  }
}

TEST_F(EagerViewOpsTest, DISABLED_AddMmOnView1) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({1000});
    torch::Tensor B = torch::randn({256, 2048, 1, 1});
    torch::Tensor C = torch::randn({1000, 2048});

    auto B_view = B.view({256, 2048});
    auto C_t = torch::t(C);

    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hC = C.to(torch::kHPU);
    auto hB_view = hB.view({256, 2048});
    auto hC_t = torch::t(hC);
    torch::Tensor O = torch::addmm(hA, hB_view, hC_t, 1, 1);

    auto computed = O.to(torch::kCPU);
    auto expected = torch::addmm(A, B_view, C_t, 1, 1);

    EXPECT_TRUE(allclose(expected, computed, 0.001, 0.001));
  }
}

TEST_F(EagerViewOpsTest, DISABLED_AddMmOnView2) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({1000});
    torch::Tensor B = torch::randn({256, 2048, 1, 1});
    torch::Tensor C = torch::randn({1000, 2048});

    auto B_view = B.view({256, 2048});
    auto C_t = torch::t(C);

    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hC = C.to(torch::kHPU);
    auto hB_view = hB.view({256, 2048});
    auto hC_t = torch::t(hC);
    torch::Tensor O = torch::addmm(hA, hB_view, hC_t, 1, 1);

    auto computed = O.to(torch::kCPU);
    auto expected = torch::addmm(A, B_view, C_t, 1, 1);

    EXPECT_TRUE(allclose(expected, computed, 0.001, 0.001));

    O = torch::addmm(hA, hB_view, hC_t, 1, 1);
    computed = O.to(torch::kCPU);
    EXPECT_TRUE(allclose(expected, computed, 0.001, 0.001));
  }
}

TEST_F(EagerViewOpsTest, DISABLED_AddMmInPlace1) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({256, 1000});
    torch::Tensor B = torch::randn({256, 2048, 1, 1});
    torch::Tensor C = torch::randn({1000, 2048});

    auto B_view = B.view({256, 2048});
    auto C_t = torch::t(C);

    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hC = C.to(torch::kHPU);
    auto hB_view = hB.view({256, 2048});
    auto hC_t = torch::t(hC);
    hA.addmm_(hB_view, hC_t, 1, 1);

    A.addmm_(B_view, C_t, 1, 1);

    EXPECT_TRUE(allclose(A, hA.cpu(), 0.001, 0.001));
  }
}

TEST_F(EagerViewOpsTest, DISABLED_AddMmInPlace2) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({256, 1000});
    torch::Tensor B = torch::randn({256, 2048, 1, 1});
    torch::Tensor C = torch::randn({1000, 2048});

    auto B_view = B.view({256, 2048});
    auto C_t = torch::t(C);

    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hC = C.to(torch::kHPU);
    auto hB_view = hB.view({256, 2048});
    auto hC_t = torch::t(hC);
    hA.addmm_(hB_view, hC_t, 1, 1);

    A.addmm_(B_view, C_t, 1, 1);

    EXPECT_TRUE(allclose(A, hA.cpu(), 0.001, 0.001));

    torch::Tensor A_2 = torch::randn({256, 1000});
    torch::Tensor B_2 = torch::randn({256, 2048, 1, 1});
    torch::Tensor C_2 = torch::randn({1000, 2048});

    auto B_2_view = B_2.view({256, 2048});
    auto C_2_t = torch::t(C_2);

    torch::Tensor hA_2 = A_2.to(torch::kHPU);
    torch::Tensor hB_2 = B_2.to(torch::kHPU);
    torch::Tensor hC_2 = C_2.to(torch::kHPU);
    auto hB_2_view = hB_2.view({256, 2048});
    auto hC_2_t = torch::t(hC_2);
    hA_2.addmm_(hB_2_view, hC_2_t, 1, 1);

    A_2.addmm_(B_2_view, C_2_t, 1, 1);

    EXPECT_TRUE(allclose(A_2, hA_2.cpu(), 0.001, 0.001));
  }
}

TEST_F(EagerViewOpsTest, AddOnAsStridedInPlace1) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({3, 3});
    auto hA = A.to(torch::kHPU);
    auto B = A.as_strided({2, 2}, {1, 2}, 1);
    B = B.add_(1.0);

    auto hB = hA.as_strided({2, 2}, {1, 2}, 1);

    // To support only as_strided, either we lower it from copy or do it on cpu
    // EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);

    hB = hB.add_(1.0);

    EXPECT_EQ(allclose(A, hA.cpu(), 0.001, 0.001), true);
  }
}

TEST_F(EagerViewOpsTest, AddOnAsStridedInPlace2) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({3, 3});
    auto hA = A.to(torch::kHPU);
    auto B = A.as_strided({2, 2}, {1, 2}, 1);
    B = B.add_(1.0);

    auto hB = hA.as_strided({2, 2}, {1, 2}, 1);

    // To support only as_strided, either we lower it from copy or do it on cpu
    // EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);

    hB = hB.add_(1.0);

    EXPECT_EQ(allclose(A, hA.cpu(), 0.001, 0.001), true);

    torch::Tensor C = torch::randn({3, 3});
    auto hC = C.to(torch::kHPU);
    auto D = C.as_strided({2, 2}, {1, 2}, 1);
    D = D.add_(1.0);

    auto hD = hC.as_strided({2, 2}, {1, 2}, 1);

    // To support only as_strided, either we lower it from copy or do it on cpu
    // EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);

    hD = hD.add_(1.0);

    EXPECT_EQ(allclose(C, hC.cpu(), 0.001, 0.001), true);
  }
}

TEST_F(EagerViewOpsTest, RReLUAsStridedInPlace) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    float lower = 0.1;
    float upper = 0.9;
    bool training = false;
    torch::Tensor A = torch::randn({2, 2, 2});
    torch::Tensor N = torch::randn({2, 2, 1});
    auto hA = A.to(torch::kHPU);
    auto hN = N.to(torch::kHPU);

    // run on CPU
    auto B = A.as_strided({2, 2, 1}, {4, 2, 1}, 0);
    torch::manual_seed(1234);
    habana::getDefaultHPUGenerator().set_current_seed(1234);
    torch::rrelu_with_noise_(
        B, N, lower, upper, training, at::detail::getDefaultCPUGenerator());

    // run on HPU
    auto hB = hA.as_strided({2, 2, 1}, {4, 2, 1}, 0);
    torch::manual_seed(1234);
    habana::getDefaultHPUGenerator().set_current_seed(1234);
    torch::rrelu_with_noise_(
        hB, hN, lower, upper, training, at::detail::getDefaultCPUGenerator());

    EXPECT_EQ(allclose(A, hA.cpu(), 0.001, 0.001), true);
  }
}

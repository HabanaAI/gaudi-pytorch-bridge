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
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "generated/lazy/wrap_kernels_declarations.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "utils/device_type_util.h"

using namespace habana_lazy;
using namespace at;

class LazyInferencePassTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    SetLazyMode();
    SetSeed();
    DisableCpuFallback();
    SetInferenceMode();
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
    habana_lazy::StageSubmission::getInstance().resetCurrentAccumulatedOps();
  }

  void TearDown() override {
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
    UnsetInferenceMode();
    RestoreMode();
  }
};

TEST_F(LazyInferencePassTest, linear) {
  if (isGaudi2()) {
    GTEST_SKIP() << "Test skipped on Gaudi2.";
  }
  torch::Tensor A = torch::randn({8, 4, 12, 7});
  torch::Tensor B = torch::randn({5, 7});
  torch::Tensor hA = A.to(kHPU);
  torch::Tensor hB = B.to(kHPU);
  auto expected = torch::linear(A, B);
  auto result = torch::linear(hA, hB);

  EXPECT_TRUE(allclose(expected, result, 0.001, 0.001));
}

TEST_F(LazyInferencePassTest, AddMmTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2});
  torch::Tensor C = torch::randn({2, 2});
  auto A_t = torch::t(A);

  torch::Tensor hA = A.to(kHPU);
  torch::Tensor hB = B.to(kHPU);
  torch::Tensor hC = C.to(kHPU);
  auto hA_t = torch::t(hA);
  torch::Tensor O = torch::addmm(hA_t, hB, hC, 1, 1);

  auto computed = O.to(torch::kCPU);
  auto expected = torch::addmm(A_t, B, C, 1, 1);

  EXPECT_TRUE(allclose(expected, computed, 0.001, 0.001));
}

TEST_F(LazyInferencePassTest, ConvInferenceTest) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  if (false == GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE))
    SET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE, true, 1);

  auto in = torch::randn({64, 4, 28, 28}, torch::dtype(torch::kFloat)); // nchw
  auto wt = torch::randn({5, 4, 3, 3}, torch::dtype(torch::kFloat)); // kchw
  auto exp = torch::conv2d(in, wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);

  auto h_in = in.to(torch::kHPU);
  auto h_wt = wt.to(torch::kHPU);

  torch::Tensor result =
      torch::conv2d(h_in, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE);
}
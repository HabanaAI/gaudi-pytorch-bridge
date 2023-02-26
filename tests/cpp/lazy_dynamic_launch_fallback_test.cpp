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

#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;
// In this class both the pass fallback and compilation fallback is enabled
class LazyDynamicDualFallbackTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    SetLazyMode();

    SetSeed();

    DisableCpuFallback();

    SetDynamicMode();

    EnableDynamicLaunchFallback();

    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();

    habana_lazy::StageSubmission::getInstance().resetCurrentAccumulatedOps();
  }

  void TearDown() override {
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
    UnsetDynamicMode();

    RestoreDynamicLaunchFallback();

    RestoreMode();
  }
};

// Also validates ComputeOutputShape for broadcast
TEST_F(LazyDynamicDualFallbackTest, ExpandTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  constexpr int Wmax{482}, Hmax{200};
  std::vector<int> W_in_sizes{1, Wmax, 1, Wmax, 1, Wmax};
  std::vector<int> H_in_sizes{Hmax, 1, Hmax, 1, 1, Hmax};
  for (int i = 0; i < W_in_sizes.size(); i++) {
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    int W = W_in_sizes[i];
    int H = H_in_sizes[i];

    torch::Tensor A = torch::randn({W, H}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);

    auto E = A.expand({Wmax, Hmax});
    torch::Tensor hE = hA.expand({Wmax, Hmax});

    auto cE = hE.to(torch::kCPU);
    EXPECT_EQ(allclose(cE, E), true);
  }
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// This test requires fallback
// Also validates ComputeOutputShape for broadcast
TEST_F(LazyDynamicDualFallbackTest, ExpandTest2) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  std::vector<int> W_in_sizes{754, 350, 664, 1};
  std::vector<int> H_in_sizes{2, 2, 2, 2};
  std::vector<int> W_expand_sizes{754, 350, 664, 500};
  for (int i = 0; i < W_in_sizes.size(); i++) {
    int W = W_in_sizes[i];
    int H = H_in_sizes[i];
    int W_expand = W_expand_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({W, H}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);

    torch::Tensor h_out = hA.expand({W_expand, 2});

    auto h_cout = h_out.to(torch::kCPU);
    auto cout = A.expand({W_expand, 2});

    EXPECT_EQ(allclose(h_cout, cout), true);
  }
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

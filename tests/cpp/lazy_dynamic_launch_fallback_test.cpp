/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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
    SetDynamicMode();

    habana_lazy_test::LazyTest::SetUp();
  }

  void TearDown() override {
    UnsetDynamicMode();

    habana_lazy_test::LazyTest::TearDown();
  }
};

// Also validates InferOutputMeta for broadcast
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
// Also validates InferOutputMeta for broadcast
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

TEST_F(LazyDynamicDualFallbackTest, DynamicBatchedNms) {
  torch::manual_seed(0);
  // Generate random scores for each box
  std::vector<int> num_boxes{10, 12};
  std::vector<std::vector<int>> refs{
      {7, 1, 5, 0, 6, 8, 4}, {11, 5, 4, 3, 2, 8, 1, 0, 7, 10}};
  for (int i = 0; i < num_boxes.size(); i++) {
    torch::Tensor scores = torch::rand({num_boxes[i]});
    torch::Tensor hscores = scores.to(torch::kHPU);

    // Generate boxes of random sizes
    torch::Tensor boxes = torch::rand({num_boxes[i], 4}) * 256;
    // ensure x2 > x1 and y2 > y1
    auto tlist = boxes.split(2, 1);
    tlist[1] = tlist[1] + tlist[0];
    auto new_boxes = torch::cat({tlist[0], tlist[1]}, 1);
    torch::Tensor hboxes = new_boxes.to(torch::kHPU);
    torch::Tensor classes_i = torch::rand({20}).to(torch::kHPU);
    torch::Tensor hclasses = torch::slice(classes_i, 0, 0, num_boxes[i], 1);
    auto nms_boxid = batched_nms_hpu_lazy(hboxes, hscores, hclasses, 0.2);

    auto ref_out = torch::tensor(refs[i]).to(torch::kLong);
    bool equal = ref_out.allclose(nms_boxid.to(torch::kCPU), 0, 0);
    EXPECT_EQ(equal, true);
  }
}

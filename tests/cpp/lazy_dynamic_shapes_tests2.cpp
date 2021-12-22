/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy_test_infra.h"
#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

using namespace habana_lazy;

class LazyDynamicShapesTest2 : public habana_lazy_test::LazyTest {};

TEST_F(LazyDynamicShapesTest2, SliceOnChlastInput) {
  bool refine_enabled = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
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
    HbLazyTensor::StepMarker({});
    EXPECT_EQ(allclose(B, hB.cpu()), true);
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest2, SliceOnChlast3dInput) {
  bool refine_enabled = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

  int N = 2, C = 3, D = 4, H = 5, W = 6;
  std::vector<int> in_sizes{8, 10, 12, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    torch::Tensor A = torch::randn({N, C, D, H, W})
                          .contiguous(c10::MemoryFormat::ChannelsLast3d);
    auto hA = A.to(torch::kHPU);
    auto B = torch::slice(A, 1, 1, -1, 1);
    auto hB = torch::slice(hA, 1, 1, -1, 1);
    HbLazyTensor::StepMarker({});
    EXPECT_EQ(allclose(B, hB.cpu()), true);
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest2, SelectOnChlast3dInput) {
  bool refine_enabled = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

  int N = 2, C = 3, D = 4, H = 5, W = 6;
  std::vector<int> in_sizes{8, 10, 12, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    torch::Tensor A = torch::randn({N, C, D, H, W})
                          .contiguous(c10::MemoryFormat::ChannelsLast3d);
    auto hA = A.to(torch::kHPU);
    auto B = torch::select(A, 3, 1);
    auto hB = torch::select(hA, 3, 1);
    HbLazyTensor::StepMarker({});
    EXPECT_EQ(allclose(B, hB.cpu()), true);
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest2, InplaceView) {
  bool refine_enabled = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

  int N = 2, C = 3, H = 4, W = 5;
  std::vector<int> in_sizes{8, 10, 12, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    torch::Tensor A = torch::randn({N, C, H, W});
    auto hA = A.to(torch::kHPU);
    auto B = A.view(-1);
    B.add_(0.5);
    // hpu
    auto hB = hA.view(-1);
    hB.add_(0.5);
    HbLazyTensor::StepMarker({});
    EXPECT_EQ(allclose(A, hA.cpu()), true);
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest2, InplaceViewon3d) {
  bool refine_enabled = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

  int N = 2, C = 3, D = 4, H = 5, W = 6;
  std::vector<int> in_sizes{8, 10, 12, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    // int W = in_sizes[i];
    torch::Tensor A = torch::randn({N, C, D, H, W});
    auto hA = A.to(torch::kHPU);
    auto B = A.view(-1);
    B.add_(0.5);
    // hpu
    auto hB = hA.view(-1);
    hB.add_(0.5);
    HbLazyTensor::StepMarker({});
    EXPECT_EQ(allclose(A, hA.cpu()), true);
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest2, DISABLED_InplaceViewonChlast) {
  bool refine_enabled = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

  int N = 2, C = 3, H = 4, W = 5;
  std::vector<int> in_sizes{8, 10, 12, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    torch::Tensor A =
        torch::randn({N, C, H, W}).contiguous(c10::MemoryFormat::ChannelsLast);
    auto hA = A.to(torch::kHPU);
    auto B = A.view(-1);
    B.add_(0.5);
    // hpu
    auto hB = hA.view(-1);
    hB.add_(0.5);
    HbLazyTensor::StepMarker({});
    EXPECT_EQ(allclose(A, hA.cpu()), true);
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest2, DISABLED_InplaceViewonChlast3d) {
  bool refine_enabled = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

  int N = 2, C = 3, D = 4, H = 5, W = 6;
  std::vector<int> in_sizes{8, 10, 12, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    torch::Tensor A = torch::randn({N, C, D, H, W})
                          .contiguous(c10::MemoryFormat::ChannelsLast3d);
    auto hA = A.to(torch::kHPU);
    auto B = A.view(-1);
    B.add_(0.5);
    // hpu
    auto hB = hA.view(-1);
    hB.add_(0.5);
    HbLazyTensor::StepMarker({});
    EXPECT_EQ(allclose(A, hA.cpu()), true);
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest2, DynamicShapeSimple_min_max_current) {
  bool refine_enabled = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
    setenv("PT_HPU_ENABLE_MIN_MAX_AS_CURRENT", "1", 1);
  }
  int A = 4;
  const int C = 3;
  std::vector<int> in_sizes{6, 8, 10};
  int num;

  for (int i = 0; i < in_sizes.size(); i++) {
    int B = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor c0 = torch::randn({C, B, A}, torch::requires_grad(false));
    torch::Tensor c1 = torch::randn({C, B, A}, torch::requires_grad(false));

    torch::Tensor c4 = torch::add(c0, c1);
    torch::Tensor c5 = torch::mul(c0, c1);
    torch::Tensor c6 = torch::mul(c4, c5);
    torch::Tensor c7 = torch::relu(c6);

    PT_TEST_DEBUG(
        "PTI_DBG :: c0.shape : ", c0.sizes(), " c0.strides : ", c0.strides());
    PT_TEST_DEBUG(
        "PTI_DBG :: c1.shape : ", c1.sizes(), " c1.strides : ", c1.strides());
    PT_TEST_DEBUG(
        "PTI_DBG :: c7.shape : ", c7.sizes(), " c7.strides : ", c7.strides());

    torch::Tensor h0 = c0.to(torch::kHPU);
    torch::Tensor h1 = c1.to(torch::kHPU);
    torch::Tensor h4 = torch::add(h0, h1);
    torch::Tensor h5 = torch::mul(h0, h1);
    torch::Tensor h6 = torch::mul(h4, h5);
    torch::Tensor h7 = torch::relu(h6);
    torch::Tensor h7_c = h7.to(torch::kCPU);

    PT_TEST_DEBUG(
        "PTI_DBG :: h0.shape : ", h0.sizes(), " h0.strides : ", h0.strides());
    PT_TEST_DEBUG(
        "PTI_DBG :: h1.shape : ", h1.sizes(), " h1.strides : ", h1.strides());
    PT_TEST_DEBUG(
        "PTI_DBG :: h7.shape : ", h7.sizes(), " h7.strides : ", h7.strides());

    EXPECT_EQ(allclose(c7, h7_c, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG :: TEST ", i, "  ========\n");
  }
  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
    unsetenv("PT_HPU_ENABLE_MIN_MAX_AS_CURRENT");
  }
}

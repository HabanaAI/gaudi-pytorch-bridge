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
#include "utils/device_type_util.h"

using namespace habana_lazy;

class LazyDynamicShapesTest2 : public habana_lazy_test::LazyDynamicTest {};

TEST_F(LazyDynamicShapesTest2, SliceOnChlastInput) {
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
}

TEST_F(LazyDynamicShapesTest2, SliceOnChlast3dInput) {
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
}

TEST_F(LazyDynamicShapesTest2, SelectOnChlast3dInput) {
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
}

TEST_F(LazyDynamicShapesTest2, InplaceView) {
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
}

TEST_F(LazyDynamicShapesTest2, InplaceViewon3d) {
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
}

TEST_F(LazyDynamicShapesTest2, InplaceViewonChlast) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  int N = 2, C = 3, H = 4, W = 5;
  std::vector<int> in_sizes{8, 10, 12, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    torch::Tensor A =
        torch::randn({N, C, H, W}).contiguous(c10::MemoryFormat::ChannelsLast);
    auto hA = A.to(torch::kHPU);
    auto B = A.reshape(A.sizes());
    B.add_(0.5);
    // hpu
    auto hB = hA.reshape(hA.sizes());
    hB.add_(0.5);
    HbLazyTensor::StepMarker({});
    EXPECT_EQ(allclose(B, hB.cpu()), true);
  }
}

// Enable this once SW-102924 is fixed.
TEST_F(LazyDynamicShapesTest2, DISABLED_InplaceViewonChlast3d) {
  int N = 2, C = 3, D = 4, H = 5, W = 6;
  std::vector<int> in_sizes{8, 10, 12, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    torch::Tensor A = torch::randn({N, C, D, H, W})
                          .contiguous(c10::MemoryFormat::ChannelsLast3d);
    auto hA = A.to(torch::kHPU);
    auto B = A.reshape(A.sizes());
    B.add_(0.5);
    // hpu
    auto hB = hA.view(hA.sizes());
    hB.add_(0.5);
    HbLazyTensor::StepMarker({});
    EXPECT_EQ(allclose(B, hB.cpu()), true);
  }
}

TEST_F(LazyDynamicShapesTest2, DynamicShapeSimple_min_max_current) {
  bool min_max_enabled = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_MIN_MAX_AS_CURRENT);
  if (!min_max_enabled) {
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_MIN_MAX_AS_CURRENT, "1", 1);
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
  if (!min_max_enabled) {
    UNSET_ENV_FLAG_NEW(PT_HPU_ENABLE_MIN_MAX_AS_CURRENT);
  }
}

TEST_F(LazyDynamicShapesTest2, VerifyPolicyEnum) {
  {
    SET_ENV_FLAG_NEW(PT_HPU_DYNAMIC_MAX_POLICY_ORDER, "1", 1);
    habana_helpers::DynamicBucketInfo bucket_info;
    ASSERT_EQ(
        bucket_info.GetMaxPolicy(), habana_helpers::DynamicDimsPolicy::CURRENT);
  }
  {
    SET_ENV_FLAG_NEW(PT_HPU_DYNAMIC_MAX_POLICY_ORDER, "2", 1);
    habana_helpers::DynamicBucketInfo bucket_info;
    ASSERT_EQ(
        bucket_info.GetMaxPolicy(),
        habana_helpers::DynamicDimsPolicy::CALCULATED);
  }
  {
    SET_ENV_FLAG_NEW(PT_HPU_DYNAMIC_MAX_POLICY_ORDER, "3", 1);
    habana_helpers::DynamicBucketInfo bucket_info;
    ASSERT_EQ(
        bucket_info.GetMaxPolicy(),
        habana_helpers::DynamicDimsPolicy::HISTORIC);
  }
  {
    SET_ENV_FLAG_NEW(PT_HPU_DYNAMIC_MAX_POLICY_ORDER, "4", 1);
    habana_helpers::DynamicBucketInfo bucket_info;
    ASSERT_EQ(
        bucket_info.GetMaxPolicy(),
        habana_helpers::DynamicDimsPolicy::LOCAL_HISTORIC);
  }
  {
    SET_ENV_FLAG_NEW(PT_HPU_DYNAMIC_MAX_POLICY_ORDER, "5", 1);
    habana_helpers::DynamicBucketInfo bucket_info;
    ASSERT_EQ(
        bucket_info.GetMaxPolicy(),
        habana_helpers::DynamicDimsPolicy::LOCAL_HIST_PER_TSR);
  }
  UNSET_ENV_FLAG_NEW(PT_HPU_DYNAMIC_MAX_POLICY_ORDER);
}

/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
#include "habana_lazy_test_infra.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include "backend/helpers/dynamic_bucket_info.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/env_flags.h"
#include "generated/lazy/wrap_kernels_declarations.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"
#include "utils/device_type_util.h"

using namespace habana_lazy;

// In this class both the pass fallback and compilation fallback are disabled
class LazyDynamicShapesTest : public habana_lazy_test::LazyDynamicTest {
 public:
  void DynamicShapeTest2(bool with_mark_step);
};

// Graph :
//
//     Bias1  Bias2           Data
//       \    /               |
//         Add                |
//          |-(weights)->  Convolution 3x3
//                            |
//                        Batch Norm
//                        Max Pool 2D           Bias3
//                           Relu             Broadcast
//                            |                  |
//                           Add <----------------
//                            |
//                       UpSampleNearest2d
//                            |
//                           out

TEST_F(LazyDynamicShapesTest, DynamicShapeTest) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  int kH = 3;
  int kW = 3;
  const int C = 16;
  const int N = 16;
  int H = 16;

  std::vector<int> in_sizes{16, 32, 64};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    int W = in_sizes[i];
    // weight_tensor = bias1 + bias2
    torch::Tensor bias1 =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    torch::Tensor bias2 =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    torch::Tensor h_bias1 = bias1.to(torch::kHPU);
    torch::Tensor h_bias2 = bias2.to(torch::kHPU);
    torch::Tensor weight_tensor = torch::add(bias1, bias2);
    torch::Tensor h_weight_tensor = torch::add(h_bias1, h_bias2);
    // out_conv = Conv3x3(Data, weight)
    torch::Tensor in_tensor =
        torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor h_in_tensor = in_tensor.to(torch::kHPU);
    torch::Tensor h_weight_tensor_hwck = h_weight_tensor;
    torch::Tensor h_out_conv = torch::conv2d(
        h_in_tensor, h_weight_tensor_hwck, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor out_conv = torch::conv2d(
        in_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    // bn_out = BatchNorm(out_conv)
    torch::Tensor gamma =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor beta =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor mean =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor var =
        torch::ones(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor h_gamma = gamma.to(torch::kHPU);
    torch::Tensor h_beta = beta.to(torch::kHPU);
    torch::Tensor h_mean = mean.to(torch::kHPU);
    torch::Tensor h_var = var.to(torch::kHPU);
    float mom = 0.1;
    float eps = 1e-5;
    auto h_bn_outs = torch::native_batch_norm(
        h_out_conv, h_gamma, h_beta, h_mean, h_var, false, mom, eps);
    auto bn_outs = torch::native_batch_norm(
        out_conv, gamma, beta, mean, var, false, mom, eps);
    auto h_bn_out = std::get<0>(h_bn_outs);
    auto bn_out = std::get<0>(bn_outs);
    // pool_out = MaxPool2D(bn_out)
    auto h_pool_outs = torch::max_pool2d_with_indices(
        h_bn_out, {2, 2}, {2, 2}, {0, 0}, {1, 1}, true);
    torch::Tensor h_pool_out = std::get<0>(h_pool_outs);
    torch::Tensor pool_out = torch::max_pool2d(bn_out, 2, 2);
    // relu_out = relu(pool_out)
    torch::Tensor h_relu_out = torch::relu(h_pool_out);
    torch::Tensor relu_out = torch::relu(pool_out);
    // out = add(relu_out, x)
    torch::Tensor bias3 =
        torch::randn(1, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor h_bias3 = bias3.to(torch::kHPU);
    auto h_out_add = torch::add(h_relu_out, h_bias3);
    auto out_add = torch::add(relu_out, bias3);
    // out = upsample(out_add,2)
    std::array<double, 2> scale_array = {2.0, 2.0};
    c10::ArrayRef<double> scale_factors = scale_array;
    auto h_out = torch::upsample_nearest2d(h_out_add, {}, scale_factors);
    auto out = torch::upsample_nearest2d(out_add, {}, scale_factors);
    torch::Tensor out_hpu = h_out.to(torch::kCPU);
    EXPECT_EQ(allclose(out_hpu, out, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
}

// Graph :
//
//     Bias1  Bias2           Data
//       \    /               |
//         Add                |
//          |-(weights)->  Convolution 3x3
//                            |
//                        Batch Norm
//                            |                Bias3
//                           Relu             Broadcast
//                            |                  |
//                           Add <----------------
//                            |
//                           out

TEST_F(LazyDynamicShapesTest, DynamicShapeTest2WithMarkStep) {
  DynamicShapeTest2(true);
}

TEST_F(LazyDynamicShapesTest, DynamicShapeTest2NoMarkStep) {
  DynamicShapeTest2(false);
}

void LazyDynamicShapesTest::DynamicShapeTest2(bool with_mark_step) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  const int N = 16;
  int H = 16;

  std::vector<int> in_sizes{16, 64};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    int W = in_sizes[i];
    // weight_tensor = bias1 + bias2
    torch::Tensor bias1 =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    torch::Tensor bias2 =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    torch::Tensor h_bias1 = bias1.to(torch::kHPU);
    torch::Tensor h_bias2 = bias2.to(torch::kHPU);
    torch::Tensor weight_tensor = torch::add(bias1, bias2);
    torch::Tensor h_weight_tensor = torch::add(h_bias1, h_bias2);
    // out_conv = Conv3x3(Data, weight)
    torch::Tensor in_tensor =
        torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor h_in_tensor = in_tensor.to(torch::kHPU);
    torch::Tensor h_weight_tensor_hwck = h_weight_tensor;

    torch::Tensor h_out_conv = torch::conv2d(
        h_in_tensor, h_weight_tensor_hwck, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor out_conv = torch::conv2d(
        in_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    // bn_out = BatchNorm(out_conv)
    torch::Tensor gamma =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor beta =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor mean =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor var =
        torch::ones(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor h_gamma = gamma.to(torch::kHPU);
    torch::Tensor h_beta = beta.to(torch::kHPU);
    torch::Tensor h_mean = mean.to(torch::kHPU);
    torch::Tensor h_var = var.to(torch::kHPU);
    float mom = 0.1;
    float eps = 1e-5;
    auto h_bn_outs = torch::native_batch_norm(
        h_out_conv, h_gamma, h_beta, h_mean, h_var, false, mom, eps);
    auto bn_outs = torch::native_batch_norm(
        out_conv, gamma, beta, mean, var, false, mom, eps);
    auto h_bn_out = std::get<0>(h_bn_outs);
    auto bn_out = std::get<0>(bn_outs);
    // relu_out = relu(bn_out)
    if (with_mark_step) {
      HbLazyTensor::StepMarker({});
    }
    torch::Tensor h_relu_out = torch::relu(h_bn_out);
    torch::Tensor relu_out = torch::relu(bn_out);
    // out = add(relu_out, x)
    torch::Tensor bias3 =
        torch::randn(1, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor h_bias3 = bias3.to(torch::kHPU);
    auto h_out = torch::add(h_relu_out, h_bias3);
    auto out = torch::add(relu_out, bias3);

    torch::Tensor out_hpu = h_out.to(torch::kCPU);
    EXPECT_EQ(allclose(out_hpu, out, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
}

// Graph :
//
//     Bias1  Bias2           Data
//       \    /               |
//        \  /              Reshape
//         Add                |
//          |-(weights)->  Convolution 3x3
//                            |
//                        Batch Norm
//                        Max Pool 2D           Bias3
//                           Relu             Broadcast
//                            |                  |
//                           Add <----------------
//                            |
//                       UpSampleNearest2d
//                            |
//                           out

TEST_F(LazyDynamicShapesTest, DynamicShapeTest3) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  const int N = 16;
  int H = 16;
  at::Scalar inScalar = 2.0;
  std::vector<int> in_sizes{16, 64};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    int W = in_sizes[i];
    // weight_tensor = bias1 + bias2
    torch::Tensor bias1 =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    torch::Tensor bias2 =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    torch::Tensor h_bias1 = bias1.to(torch::kHPU);
    torch::Tensor h_bias2 = bias2.to(torch::kHPU);
    torch::Tensor weight_tensor = torch::add(bias1, bias2);
    torch::Tensor h_weight_tensor = torch::add(h_bias1, h_bias2);
    // out_conv = Conv3x3(Data, weight)
    torch::Tensor in_tensor =
        torch::randn(N * C * H * W, torch::requires_grad(false));
    torch::Tensor h_in_tensor = in_tensor.to(torch::kHPU);
    torch::Tensor h_weight_tensor_hwck = h_weight_tensor;
    torch::Tensor h_out_conv = torch::conv2d(
        h_in_tensor.reshape({N, C, H, W}),
        h_weight_tensor_hwck,
        {},
        {1},
        at::IntArrayRef{0},
        {1},
        1);
    torch::Tensor out_conv = torch::conv2d(
        in_tensor.reshape({N, C, H, W}),
        weight_tensor,
        {},
        {1},
        at::IntArrayRef{0},
        {1},
        1);
    // bn_out = BatchNorm(out_conv)
    torch::Tensor gamma =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor beta =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor mean =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor var =
        torch::ones(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor h_gamma = gamma.to(torch::kHPU);
    torch::Tensor h_beta = beta.to(torch::kHPU);
    torch::Tensor h_mean = mean.to(torch::kHPU);
    torch::Tensor h_var = var.to(torch::kHPU);
    float mom = 0.1;
    float eps = 1e-5;
    auto h_bn_outs = torch::native_batch_norm(
        h_out_conv, h_gamma, h_beta, h_mean, h_var, false, mom, eps);
    auto bn_outs = torch::native_batch_norm(
        out_conv, gamma, beta, mean, var, false, mom, eps);
    auto h_bn_out = std::get<0>(h_bn_outs);
    auto bn_out = std::get<0>(bn_outs);
    // pool_out = MaxPool2D(bn_out)
    auto h_pool_outs = torch::max_pool2d_with_indices(
        h_bn_out, {2, 2}, {2, 2}, {0, 0}, {1, 1}, true);
    torch::Tensor h_pool_out = std::get<0>(h_pool_outs);
    torch::Tensor pool_out = torch::max_pool2d(bn_out, 2, 2);
    // relu_out = relu(pool_out)
    torch::Tensor h_relu_out = torch::relu(h_pool_out);
    torch::Tensor relu_out = torch::relu(pool_out);
    // out = add(relu_out, x)
    torch::Tensor bias3 =
        torch::randn(1, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor h_bias3 = bias3.to(torch::kHPU);
    auto h_out_add = torch::add(h_relu_out, h_bias3);
    auto out_add = torch::add(relu_out, bias3);
    // out = upsample(out_add,2)
    std::array<double, 2> scale_array = {2.0, 2.0};
    c10::ArrayRef<double> scale_factors = scale_array;
    auto h_out = torch::upsample_nearest2d(h_out_add, {}, scale_factors);
    auto out = torch::upsample_nearest2d(out_add, {}, scale_factors);

    torch::Tensor out_hpu = h_out.to(torch::kCPU);
    EXPECT_EQ(allclose(out_hpu, out, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
}

TEST_F(LazyDynamicShapesTest, DynamicShapeDebugSimple) {
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
}

TEST_F(LazyDynamicShapesTest, DynamicShapeClearCachedRecipes) {
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

    habana::ClearDynamicBucketRecipeInfo();

    EXPECT_EQ(habana::RecipeCacheLRU::get_cache().empty(), true);
    EXPECT_EQ(habana::DynamicBucketInfoMap::get_instance().empty(), true);

    PT_TEST_DEBUG("PTI_DBG :: TEST ", i, "  ========\n");
  }
}

TEST_F(LazyDynamicShapesTest, RefineSlice) {
  int N = 1;
  int C = 4;
  int H = 24;
  std::vector<int> W_values{16, 36};
  std::vector<int> rounds{1, 6};
  for (int i = 0; i < W_values.size(); i++) {
    for (int j = 1; j <= rounds[i]; j++) {
      PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i + 1, ", round ", j, "  START");
      int W = W_values[i];
      torch::Tensor A = torch::randn({N, C, H, W}, torch::requires_grad(false));
      torch::Tensor hA = A.to(torch::kHPU);
      int64_t dim = 2;
      int64_t start_index = 0;
      int64_t end = 3;
      int64_t step = 1;

      torch::Tensor h_out = torch::slice(hA, dim, start_index, end, step);
      HbLazyTensor::StepMarker({});
      auto h_cout = h_out.to(torch::kCPU);
      PT_TEST_DEBUG("PTI_DBG :: TEST ", i + 1, ", round ", j, "  END");
    }
  }
}

TEST_F(LazyDynamicShapesTest, SingleOpRelu) {
  int A = 4;
  const int C = 3;
  std::vector<int> in_sizes{6, 8, 10, 12, 14, 16};
  int num;

  int rounds{10};
  while (rounds--) {
    PT_TEST_DEBUG("\nPTI_DBG :: round ", rounds, "  --------\n");
    for (int i = 0; i < in_sizes.size(); i++) {
      int B = in_sizes[i];
      PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------");
      torch::Tensor c0 = torch::randn({C, B, A}, torch::requires_grad(false));

      torch::Tensor c4 = torch::relu(c0);

      PT_TEST_DEBUG(
          "PTI_DBG :: c0.shape : ", c0.sizes(), " c0.strides : ", c0.strides());
      PT_TEST_DEBUG(
          "PTI_DBG :: c1.shape : ", c4.sizes(), " c4.strides : ", c4.strides());

      torch::Tensor h0 = c0.to(torch::kHPU);
      torch::Tensor h4 = torch::relu(h0);
      torch::Tensor h4_c = h4.to(torch::kCPU);

      PT_TEST_DEBUG(
          "PTI_DBG :: h0.shape : ", h0.sizes(), " h0.strides : ", h0.strides());
      PT_TEST_DEBUG(
          "PTI_DBG :: h1.shape : ", h4.sizes(), " h4.strides : ", h4.strides());

      EXPECT_EQ(allclose(c4, h4_c, 0.01, 0.01), true);
      PT_TEST_DEBUG("PTI_DBG :: TEST ", i, "  ========");
    }
  }
}

// Reproducer for https://jira.habana-labs.com/browse/SW-94443
TEST_F(LazyDynamicShapesTest, SingleOpAdd) {
  std::vector<std::vector<int64_t>> in1 = {{10, 20, 30}, {10, 50, 30}};
  std::vector<std::vector<int64_t>> in2 = {{10, 1, 1}, {10, 50, 30}};

  for (size_t i = 0; i < in1.size(); i++) {
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------");
    torch::Tensor c0 = torch::randn(in1[i], torch::requires_grad(false));
    torch::Tensor c1 = torch::randn(in2[i], torch::requires_grad(false));

    torch::Tensor c4 = torch::add(c0, c1);

    PT_TEST_DEBUG(
        "PTI_DBG :: c0.shape : ", c0.sizes(), " c0.strides : ", c0.strides());
    PT_TEST_DEBUG(
        "PTI_DBG :: c1.shape : ", c1.sizes(), " c1.strides : ", c1.strides());
    PT_TEST_DEBUG(
        "PTI_DBG :: c4.shape : ", c4.sizes(), " c4.strides : ", c4.strides());

    torch::Tensor h0 = c0.to(torch::kHPU);
    torch::Tensor h1 = c1.to(torch::kHPU);
    torch::Tensor h4 = torch::add(h0, h1);
    torch::Tensor h4_c = h4.to(torch::kCPU);

    PT_TEST_DEBUG(
        "PTI_DBG :: h0.shape : ", h0.sizes(), " h0.strides : ", h0.strides());
    PT_TEST_DEBUG(
        "PTI_DBG :: h1.shape : ", h1.sizes(), " h1.strides : ", h1.strides());
    PT_TEST_DEBUG(
        "PTI_DBG :: h4.shape : ", h4.sizes(), " h4.strides : ", h4.strides());

    EXPECT_EQ(allclose(c4, h4_c, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG :: TEST ", i, "  ========");
  }
}

TEST_F(LazyDynamicShapesTest, SetDynamicModeTest_UniqueGraph) {
  std::vector<std::pair<int, int>> v = {
      {3, 1},
      {1, 1},
      {4, 1},
      {1, 1},
      {6, 1},
      {1, 1},
      {2, 1},
      {2, 1},
      {3, 1},
      {3, 1},
      {4, 2},
      {4, 2},
      {3, 2},
      {3, 2},
      {5, 3},
      {5, 3}};
  for (std::vector<std::pair<int, int>>::iterator it = std::begin(v);
       it != std::end(v);) {
    HbLazyTensor::IterStepMarker();
    auto size1 = *it++;
    auto in1 = torch::randn(
        {size1.first, size1.second},
        torch::dtype(torch::kFloat).requires_grad(false));
    auto size2 = *it++;
    auto in2 = torch::randn(
        {size2.first, size2.second},
        torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor h_in1 = in1.to(torch::kHPU);
    torch::Tensor h_in2 = in2.to(torch::kHPU);
    auto add_1 = torch::add(h_in1, h_in2);
    torch::Tensor cpu_add_1 = add_1.to(torch::kCPU);
  }
}

TEST_F(LazyDynamicShapesTest, UniqueGraph_Broadcast) {
  // unset the env variable if set for this case
  bool org_state = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_BROADCAST_BUCKET_HANDLING);
  SET_ENV_FLAG_NEW(PT_HPU_ENABLE_BROADCAST_BUCKET_HANDLING, false, 1);
  std::pair<int, int> tensor0_sizes = {3, 3};
  std::vector<std::pair<int, int>> addSizes = {{3, 3}, {3, 1}, {1, 3}, {1, 1}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < addSizes.size(); j++) {
      // HbLazyTensor::IterStepMarker();
      int H1 = addSizes[j].first;
      int W1 = addSizes[j].second;
      if (addSizes[j].first != 1) {
        H1 = addSizes[j].first + i;
      }
      if (addSizes[j].second != 1) {
        W1 = addSizes[j].second + i;
      }
      auto in1 = torch::randn(
          {H1, W1}, torch::dtype(torch::kFloat).requires_grad(false));
      auto in2 = torch::randn(
          {tensor0_sizes.first + i, tensor0_sizes.second + i},
          torch::dtype(torch::kFloat).requires_grad(false));
      torch::Tensor h_in1 = in1.to(torch::kHPU);
      torch::Tensor h_in2 = in2.to(torch::kHPU);
      auto add_1 = torch::add(h_in1, h_in2);
      torch::Tensor cpu_add_1 = add_1.to(torch::kCPU);
    }
  }
  SET_ENV_FLAG_NEW(PT_HPU_ENABLE_BROADCAST_BUCKET_HANDLING, org_state, 1);
}

TEST_F(LazyDynamicShapesTest, SetDynamicModeTest1) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  // Check org state of env flag
  bool refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  // set the env variable false if set true for this case
  bool org_state = refine_enabled;
  if (refine_enabled) {
    habana_helpers::DisableRefineDynamicShape();
  }
  refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  ASSERT_EQ(refine_enabled, false);
  HbLazyTensor::SetDynamicMode();
  // Check if env flag set for op Accumulation/execution
  refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  ASSERT_EQ(refine_enabled, true);
  torch::Tensor num1 =
      torch::randn({C, C, kW, kH}, torch::requires_grad(false));
  torch::Tensor num2 =
      torch::randn({C, C, kW, kH}, torch::requires_grad(false));
  torch::Tensor h_num1 = num1.to(torch::kHPU);
  torch::Tensor h_num2 = num2.to(torch::kHPU);
  torch::Tensor sum_tensor = torch::add(h_num1, h_num2);
  torch::Tensor sum_cpu = torch::add(num1, num2);
  HbLazyTensor::StepMarker({});
  // Check if env flag is unset after execution
  refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  ASSERT_EQ(refine_enabled, false);
  torch::Tensor sum_hpu = sum_tensor.to(torch::kCPU);
  EXPECT_EQ(allclose(sum_cpu, sum_hpu, 0.01, 0.01), true);
  // restore the original state
  habana_helpers::SetRefineDynamicShape(org_state);
}

TEST_F(LazyDynamicShapesTest, SetDynamicModeTest2) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  // Set the env flag if not set
  bool refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  bool org_state = refine_enabled;
  if (!refine_enabled) {
    habana_helpers::EnableRefineDynamicShape();
  }
  // Check org state of env flag
  refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  ASSERT_EQ(refine_enabled, true);
  HbLazyTensor::SetDynamicMode();
  // Check if env flag set for op Accumulation/execution
  refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  ASSERT_EQ(refine_enabled, true);
  torch::Tensor num1 =
      torch::randn({C, C, kW, kH}, torch::requires_grad(false));
  torch::Tensor num2 =
      torch::randn({C, C, kW, kH}, torch::requires_grad(false));
  torch::Tensor h_num1 = num1.to(torch::kHPU);
  torch::Tensor h_num2 = num2.to(torch::kHPU);
  torch::Tensor sum_tensor = torch::add(h_num1, h_num2);
  torch::Tensor sum_cpu = torch::add(num1, num2);
  HbLazyTensor::StepMarker({});
  // Check if env flag is same after execution(since set through
  // env and not through SetDynamicMode)
  refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  ASSERT_EQ(refine_enabled, true);
  torch::Tensor sum_hpu = sum_tensor.to(torch::kCPU);
  EXPECT_EQ(allclose(sum_cpu, sum_hpu, 0.01, 0.01), true);
  // restore the original env variable
  if (!org_state) {
    habana_helpers::DisableRefineDynamicShape();
  }
}

TEST_F(LazyDynamicShapesTest, SetDynamicModeTest3) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  // Check org state of env flag
  bool refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  // set the env variable false if set true for this case
  bool org_state = refine_enabled;
  if (refine_enabled) {
    habana_helpers::DisableRefineDynamicShape();
  }
  // Check if dynamic mode is unset
  refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  ASSERT_EQ(refine_enabled, false);
  torch::Tensor num1 =
      torch::randn({C, C, kW, kH}, torch::requires_grad(false));
  torch::Tensor num2 =
      torch::randn({C, C, kW, kH}, torch::requires_grad(false));
  torch::Tensor h_num1 = num1.to(torch::kHPU);
  torch::Tensor h_num2 = num2.to(torch::kHPU);
  torch::Tensor sum_tensor = torch::add(h_num1, h_num2);
  torch::Tensor sum_cpu = torch::add(num1, num2);
  HbLazyTensor::StepMarker({});
  // Check if env flag is still unset
  refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  ASSERT_EQ(refine_enabled, false);
  torch::Tensor sum_hpu = sum_tensor.to(torch::kCPU);
  EXPECT_EQ(allclose(sum_cpu, sum_hpu, 0.01, 0.01), true);
  // restore the original state
  habana_helpers::SetRefineDynamicShape(org_state);
}

TEST_F(LazyDynamicShapesTest, ProdTest) {
  int H = 4;
  std::vector<int> in_sizes{6, 8, 10};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hOut = torch::prod(hA);
    torch::Tensor Out = torch::prod(A);
    EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.001, 0.001), true);
  }
}

TEST_F(LazyDynamicShapesTest, ProdDimIntTest) {
  auto prod_test = [](std::vector<int64_t> input_shape,
                      int64_t dim,
                      bool keepdim,
                      c10::ScalarType dtype) {
    auto input = torch::randn(input_shape);
    auto input_h = input.to(torch::kHPU);

    torch::Tensor hOut = at::empty_like(input_h, dtype);
    torch::Tensor out_cpu = at::empty_like(input, dtype);
    torch::prod_outf(input, dim, keepdim, dtype, out_cpu);
    torch::prod_outf(input_h, dim, keepdim, dtype, hOut);

    EXPECT_EQ(allclose(hOut.to(torch::kCPU), out_cpu, 0.001, 0.001), true);
  };

  prod_test({3, 5, 2, 3}, /*dim*/ 1, /*keepdim*/ true, /*dtype*/ torch::kFloat);
  prod_test({3, 5, 2, 8}, /*dim*/ 1, /*keepdim*/ true, /*dtype*/ torch::kFloat);
  prod_test(
      {3, 5, 2, 11}, /*dim*/ 1, /*keepdim*/ true, /*dtype*/ torch::kFloat);
}

TEST_F(LazyDynamicShapesTest, AllDimTest) {
  auto all_test =
      [](std::vector<int64_t> input_shape, int64_t dim, bool keepdim) {
        torch::ScalarType dtype = torch::kBool;
        auto input = torch::randn(input_shape).to(dtype);
        auto input_h = input.to(torch::kHPU);

        torch::Tensor hOut = at::empty_like(input_h);
        torch::Tensor out_cpu = at::empty_like(input);
        torch::all_out(out_cpu, input, dim, keepdim);
        torch::all_out(hOut, input_h, dim, keepdim);

        EXPECT_EQ(allclose(hOut.to(torch::kCPU), out_cpu, 0.001, 0.001), true);
      };

  all_test({12, 32}, /*dim*/ 0, /*keepdim*/ true);
  all_test({12, 40}, /*dim*/ 0, /*keepdim*/ true);
  all_test({12, 45}, /*dim*/ 0, /*keepdim*/ true);
}

TEST_F(LazyDynamicShapesTest, SliceTest) {
  int N = 1;
  int C = 4;
  int H = 24;
  std::vector<int> W_values{16, 18, 20, 36};
  std::vector<int> in_start{0, 2, 3, 28};
  std::vector<int> in_end{8, 10, 12, 36};
  std::vector<int> in_step{1, 1, 1, 1};
  for (int i = 0; i < W_values.size(); i++) {
    int W = W_values[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    int64_t dim = 3;
    int64_t start_index = in_start[i];
    int64_t end = in_end[i];
    int64_t step = in_step[i];
    torch::Tensor h_out = torch::slice(hA, dim, start_index, end, step);
    auto h_cout = h_out.to(torch::kCPU);
  }
}

TEST_F(LazyDynamicShapesTest, SliceTestUpdateBucket) {
  int H = 24;
  std::vector<int> W_values{16, 18, 20};
  std::vector<int> in_start{0, 2, 3};
  std::vector<int> in_end{8, 10, 12};
  std::vector<int> in_step{1, 1, 1};
  for (int i = 0; i < W_values.size(); i++) {
    int W = W_values[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    int64_t dim = 0;
    int64_t start_index = in_start[i];
    int64_t end = in_end[i];
    int64_t step = in_step[i];
    torch::Tensor h_out = torch::slice(hA, dim, start_index, end, step);
    auto h_cout = h_out.to(torch::kCPU);
  }
}

TEST_F(LazyDynamicShapesTest, SliceTestUpdateBucketWithNodes) {
  std::vector<int> W_values{16, 18, 20};
  std::vector<int> in_start{0, 2, 3};
  std::vector<int> in_end{8, 10, 12};
  std::vector<int> in_step{1, 1, 1};
  for (int i = 0; i < W_values.size(); i++) {
    int W = W_values[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    auto hb = hA + 1.0;
    hb = torch::relu(hb);
    hA = hA + hb;
    int64_t dim = 0;
    int64_t start_index = in_start[i];
    int64_t end = in_end[i];
    int64_t step = in_step[i];
    torch::Tensor h_out = torch::slice(hA, dim, start_index, end, step);
    auto h_cout = h_out.to(torch::kCPU);
  }
}

TEST_F(LazyDynamicShapesTest, DISABLED_SliceTest6dim) {
  int N = 1;
  int C = 4;
  int H = 24;
  int dim5 = 16;
  int dim6 = 16;
  std::vector<int> W_values{16, 18, 20};
  std::vector<int> in_start{0, 2, 3};
  std::vector<int> in_end{8, 10, 12};
  std::vector<int> in_step{1, 1, 1};
  for (int i = 0; i < W_values.size(); i++) {
    int W = W_values[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A =
        torch::randn({N, C, H, W, dim5, dim6}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    int64_t dim = 3;
    int64_t start_index = in_start[i];
    int64_t end = in_end[i];
    int64_t step = in_step[i];
    torch::Tensor h_out = torch::slice(hA, dim, start_index, end, step);
    auto h_cout = h_out.to(torch::kCPU);
  }
}

TEST_F(LazyDynamicShapesTest, SliceTest2) {
  int H = 4;
  std::vector<int> in_sizes{16, 18, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    int64_t dim = 1;
    int64_t start_index = 4;
    int64_t end = 9223372036854775807;
    int64_t step = 2;

    torch::Tensor h_out = torch::slice(hA, dim, start_index, end, step);

    auto h_cout = h_out.to(torch::kCPU);
    auto cout = torch::slice(A, dim, start_index, end, step);

    EXPECT_EQ(allclose(h_cout, cout), true);
  }
}

TEST_F(LazyDynamicShapesTest, SliceTest3) {
  int N = 1;
  int C = 4;
  int H = 4;

  std::vector<int> in_sizes{16, 18, 20, 18, 20};
  std::vector<int> in_start{0, 1, 3, 1, 2};
  std::vector<int> in_step{1, 1, 1, 1, 1};
  std::vector<int> in_end{8, 8, 12, 8, 8};

  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    int64_t dim = 3;
    int64_t start_index = in_start[i];
    int64_t end = in_end[i];
    int64_t step = in_step[i];
    torch::Tensor h_out = torch::slice(hA, dim, start_index, end, step);

    auto h_cout = h_out.to(torch::kCPU);
    auto cout = torch::slice(A, dim, start_index, end, step);

    EXPECT_EQ(allclose(h_cout, cout), true);
  }
}

TEST_F(LazyDynamicShapesTest, RepeatTest) {
  int H = 4;
  std::vector<int> in_sizes{10, 231, 520};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);

    torch::Tensor h_out = hA.repeat({5, 1, 1});

    auto h_cout = h_out.to(torch::kCPU);
    auto cout = A.repeat({5, 1, 1});

    EXPECT_EQ(allclose(h_cout, cout), true);
  }
}

TEST_F(LazyDynamicShapesTest, RepeatTest2) {
  int H = 4;
  std::vector<int> in_sizes{10, 231, 520, 600};
  std::vector<std::vector<int64_t>> repeat_sizes{
      {5, 1, 3}, {20, 1, 3}, {10, 1, 3}, {15, 1, 3}};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    auto repeatIndices = c10::IntArrayRef(repeat_sizes[i]);
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);

    torch::Tensor h_out = hA.repeat(repeatIndices);

    auto h_cout = h_out.to(torch::kCPU);
    auto cout = A.repeat(repeatIndices);

    EXPECT_EQ(allclose(h_cout, cout), true);
  }
}

void runWeightNormTest() {
  std::vector<int> in_sizes{64, 32};
  for (int i = 0; i < in_sizes.size(); i++) {
    at::Tensor v_in = at::randn({512, 32, in_sizes[i]});
    at::Tensor g_in = at::randn({1, 1, in_sizes[i]});
    int64_t dim(2);

    at::Tensor output = at::_weight_norm(v_in, g_in, dim);

    at::Tensor h_v_in = v_in.to(at::device(at::kHPU));
    at::Tensor h_g_in = g_in.to(at::device(at::kHPU));
    at::Tensor h_output = at::_weight_norm(h_v_in, h_g_in, dim);

    at::Tensor h_output_cpu = h_output.to(at::device(at::kCPU));
    EXPECT_EQ(allclose(h_output_cpu, output, 0.0001), true);
  }
}
TEST_F(LazyDynamicShapesTest, WeightNormTest) {
  const char* recipe_cache_path = GET_ENV_FLAG_NEW(PT_RECIPE_CACHE_PATH);
  SET_ENV_FLAG_NEW(PT_RECIPE_CACHE_PATH, "/tmp/WeightNormTest_dumps", 1);
  runWeightNormTest();

  // Rerun using disk caching
  runWeightNormTest();
  SET_ENV_FLAG_NEW(PT_RECIPE_CACHE_PATH, recipe_cache_path, 1);
}

TEST_F(LazyDynamicShapesTest, DynamicShapeInplaceTest) {
  int A = 2;
  std::vector<int> in_sizes{2, 3, 4};
  int num;

  for (int i = 0; i < in_sizes.size(); i++) {
    int B = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor c0 = torch::randn({A, B});
    torch::Tensor c1 = torch::randn({A, B});
    torch::Tensor c2 = torch::randn({A, B});

    torch::Tensor h0 = c0.to(torch::kHPU);
    torch::Tensor h1 = c1.to(torch::kHPU);
    torch::Tensor h2 = c2.to(torch::kHPU);

    c0 = c0.add_(c1);
    auto c3 = torch::mul(c0, c2);

    h0 = h0.add_(h1);
    torch::Tensor h3 = torch::mul(h0, h2);
    torch::Tensor h3_c = h3.to(torch::kCPU);

    EXPECT_EQ(allclose(c3, h3_c, 0.01, 0.01), true);
  }
}

TEST_F(LazyDynamicShapesTest, DISABLED_DynamicShapeInplaceTest2) {
  int A = 2;
  std::vector<int> in_sizes{2, 3, 4};
  int num;

  for (int i = 0; i < in_sizes.size(); i++) {
    int B = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor c0 = torch::randn({A, B});
    torch::Tensor c1 = torch::randn({A, B});
    torch::Tensor c2 = torch::randn({A, B});

    torch::Tensor h0 = c0.to(torch::kHPU);
    torch::Tensor h1 = c1.to(torch::kHPU);
    torch::Tensor h2 = c2.to(torch::kHPU);

    auto c3 = torch::relu(c0);
    c3 = c3.add_(c1);
    auto c4 = torch::mul(c3, c2);

    auto h3 = torch::relu(h0);
    h3 = h3.add_(h1);
    torch::Tensor h4 = torch::mul(h3, h2);
    torch::Tensor h4_c = h4.to(torch::kCPU);

    EXPECT_EQ(allclose(c4, h4_c, 0.01, 0.01), true);
  }
}

TEST_F(LazyDynamicShapesTest, DynamicShapeInplaceReluTest) {
  int A = 1;
  const int C = 1;
  std::vector<int> in_sizes{2, 4, 8};
  int num;

  for (int i = 0; i < in_sizes.size(); i++) {
    int B = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor c0 = torch::randn({C, B, A}, torch::requires_grad(false));

    c0 = torch::relu_(c0);

    torch::Tensor h0 = c0.to(torch::kHPU);

    h0 = torch::relu_(h0);
    torch::Tensor h0_c = h0.to(torch::kCPU);

    EXPECT_EQ(allclose(c0, h0_c, 0.01, 0.01), true);
  }
}

TEST_F(LazyDynamicShapesTest, AddConstantTest) {
  // test case for result = add(tensor, scalar, alpha)
  int N = 1;
  int C = 4;
  int H = 4;
  at::Scalar B = 2.0;
  at::Scalar alpha = 1.0;
  std::vector<int> in_sizes{16, 18, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor out_hpu = torch::add(hA, B, alpha);
    torch::Tensor out_cpu = torch::add(A, B, alpha);
    auto out = out_hpu.to(torch::kCPU);
    EXPECT_EQ(allclose(out, out_cpu, 0.001, 0.001), true);
  }
}

TEST_F(LazyDynamicShapesTest, AddViewTest) {
  int N = 2;
  int C = 4;
  int H = 4;
  at::Scalar alpha = 1.0;
  at::Scalar Y = 2.0;
  std::vector<int> in_sizes{16, 18, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor B = torch::randn({C, H, N}, torch::requires_grad(false));
    torch::Tensor hB = B.to(torch::kHPU);
    std::vector<int64_t> shape{N, C, H, 1};
    auto Z = torch::add(B, Y, alpha);
    auto hZ = torch::add(hB, Y, alpha);
    torch::Tensor C = Z.reshape(c10::IntArrayRef(shape));
    torch::Tensor hC = hZ.reshape(c10::IntArrayRef(shape));
    torch::Tensor out_hpu = torch::add(hA, hC, alpha);
    torch::Tensor out_cpu = torch::add(A, C, alpha);
    auto out = out_hpu.to(torch::kCPU);
    EXPECT_EQ(allclose(out, out_cpu, 0.001, 0.001), true);
  }
}

TEST_F(LazyDynamicShapesTest, CastTest) {
  PT_TEST_DEBUG("\nPTI_DBG :: TEST ", 0, "  --------\n");
  torch::Tensor A = torch::randn({1}, torch::dtype(torch::kBFloat16));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = hA.to(torch::kFloat);
  torch::Tensor Out = A.to(torch::kFloat);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.001, 0.001), true);
  int H = 512;
  std::vector<int> in_sizes{2048, 4096};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i + 1, "  --------\n");
    torch::Tensor A = torch::randn({W, H}, torch::dtype(torch::kBFloat16));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hOut = hA.to(torch::kFloat);
    torch::Tensor Out = A.to(torch::kFloat);
    EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.001, 0.001), true);
  }
}

TEST_F(LazyDynamicShapesTest, UniqueOp) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  c10::ScalarType dtype{torch::kInt32};

  std::vector<int> in_sizes{4, 6, 8};
  for (int i = 0; i < in_sizes.size(); i++) {
    int H = 4;
    int W = in_sizes[i];
    torch::Tensor input_cpu = torch::randint(1, 9, {H, W}, dtype);
    torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
    auto out_hpu = std::get<0>(torch::_unique2(input_hpu, false, false, false));
    auto out_cpu = std::get<0>(torch::_unique2(input_cpu, false, false, false));
    PRINT_TENSOR_WITH_DATA(out_cpu);
    auto h_cout = out_hpu.to(torch::kCPU);
    EXPECT_EQ(
        allclose(
            std::get<0>(h_cout.view(-1).sort()),
            std::get<0>(out_cpu.view(-1).sort())),
        true);

    auto out_cpuv = std::get<0>(out_cpu.view(-1).sort());
    auto out_hpuv = std::get<0>(h_cout.view(-1).sort());
  }
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyDynamicShapesTest, SingleOpNonzero) {
  int A = 8;
  const int RMIN = 0;
  const int RMAX = 10;
  std::vector<int> in_sizes{6, 8, 10};
  int num;

  for (int i = 0; i < in_sizes.size(); i++) {
    int B = in_sizes[i];
    PT_TEST_DEBUG("TEST ", i, "  --------");
    torch::Tensor c0 =
        torch::randint(RMIN, RMAX, {A, B}, torch::dtype(torch::kInt64));

    torch::Tensor out_cpu = torch::nonzero(c0);

    PRINT_TENSOR_WITH_DATA(c0);
    PRINT_TENSOR_WITH_DATA(out_cpu);

    torch::Tensor h0 = c0.to(torch::kHPU);
    torch::Tensor out_hpu = torch::nonzero(h0);
    torch::Tensor out_hpu_c = out_hpu.to(torch::kCPU);

    PRINT_TENSOR_WITH_DATA(h0);
    PRINT_TENSOR_WITH_DATA(out_hpu_c);

    EXPECT_EQ(allclose(out_cpu, out_hpu_c, 0.01, 0.01), true);
    PT_TEST_DEBUG("TEST ", i, "  ========");
  }
}

#define X0 0
#define Y0 1
#define X1 2
#define Y1 3

void compute_iou(
    torch::Tensor& boxes,
    std::vector<std::vector<float>>& iou_vec_2d) {
  TORCH_CHECK(boxes.dim() == 2, "Expecting a 2D tensor, got ", boxes.dim());
  TORCH_CHECK(
      boxes.sizes()[1] == 4, "Expecting the FCD=4, got ", boxes.sizes()[1]);

  auto num_boxes = boxes.sizes()[0];
  auto fcd = boxes.sizes()[1];

  PT_TEST_DEBUG(
      "PTI_DBG :: Will compute the iou for ",
      num_boxes,
      " boxes with fcd ",
      fcd);

  for (size_t i = 0; i < num_boxes - 1; i++) {
    std::vector<float> iou_vec;
    for (size_t j = i + 1; j < num_boxes; j++) {
      float iou{0.0};
      float x0i = boxes[i][X0].item<float>();
      float y0i = boxes[i][Y0].item<float>();
      float x1i = boxes[i][X1].item<float>();
      float y1i = boxes[i][Y1].item<float>();
      TORCH_CHECK(
          x0i < x1i && y0i < y1i,
          "invalid box coordinate received ",
          "  x0i=",
          x0i,
          ", y0i=",
          y0i,
          ", x1i=",
          x1i,
          ", y1i=",
          y1i);
      PT_TEST_DEBUG(
          "boxes[",
          i,
          "] :",
          "  x0i=",
          x0i,
          ", y0i=",
          y0i,
          ", x1i=",
          x1i,
          ", y1i=",
          y1i);

      float x0j = boxes[j][X0].item<float>();
      float y0j = boxes[j][Y0].item<float>();
      float x1j = boxes[j][X1].item<float>();
      float y1j = boxes[j][Y1].item<float>();
      TORCH_CHECK(
          x0j < x1j && y0j < y1j,
          "invalid box coordinate received ",
          "  x0j=",
          x0j,
          ", y0j=",
          y0j,
          ", x1j=",
          x1j,
          ", y1j=",
          y1j);
      PT_TEST_DEBUG(
          "boxes[",
          j,
          "] :",
          "  x0j=",
          x0j,
          ", y0j=",
          y0j,
          ", x1j=",
          x1j,
          ", y1j=",
          y1j);
      auto x_l = std::max(x0i, x0j);
      auto y_b = std::max(y0i, y0j);
      auto x_r = std::min(x1i, x1j);
      auto y_t = std::min(y1i, y1j);

      // Check whether boxes[i] and boxes[j] has an overlap
      if (x_l < x_r && y_b < y_t) {
        auto i_area = (x_r - x_l) * (y_t - y_b);
        auto boxi_area = (x1i - x0i) * (y1i - y0i);
        auto boxj_area = (x1j - x0j) * (y1j - y0j);
        auto u_area = boxi_area + boxj_area - i_area;
        iou = i_area / u_area;
      }

      iou_vec.push_back(iou);
      PT_TEST_DEBUG("iou of ", i, " and ", j, '=', iou);
    }
    PT_TEST_DEBUG("iou_vec[", i, "] : ", iou_vec);
    iou_vec_2d.push_back(iou_vec);
  }
}

TEST_F(LazyDynamicShapesTest, NmsSmallRef) {
  torch::manual_seed(0);

  auto num_boxes = 10;
  auto num_boxes_fixed = 8;
  auto num_boxes_variable = num_boxes - num_boxes_fixed;
  torch::Tensor scores = torch::rand({num_boxes});
  torch::Tensor boxes_fixed = torch::rand({num_boxes_fixed, 4}) * 256;

  auto num_expected_boxes{num_boxes};
  if (num_expected_boxes) {
    torch::Tensor hscores = scores.to(torch::kHPU);

    // Generate boxes of random sizes
    torch::Tensor boxes_variable = torch::rand({num_boxes_variable, 4}) * 256;
    torch::Tensor boxes = torch::cat({boxes_fixed, boxes_variable}, 0);

    // Ensure x1 > x0 and y1 > y0
    auto tlist = boxes.split(2, 1);
    tlist[1] = tlist[1] + tlist[0];
    auto valid_boxes = torch::cat({tlist[0], tlist[1]}, 1);
    // PRINT_TENSOR_WITH_DATA(valid_boxes);

    // Compute the iou scores
    // std::vector<std::vector<float>> iou_vec_2d;
    // iou_vec_2d.reserve(num_boxes-1);
    // compute_iou(valid_boxes, iou_vec_2d);

    torch::Tensor hboxes = valid_boxes.to(torch::kHPU);

    auto nms_boxid = torchvision_nms_hpu_wrap(hboxes, hscores, 1.0);
    auto nms_boxid_c = nms_boxid.to(torch::kCPU);
    TORCH_CHECK(
        nms_boxid_c.dim() == 1,
        "Expecting a 1D tensor, got ",
        boxes.dim(),
        "D tensor");
    PT_TEST_DEBUG(
        "num_expected_boxes=",
        num_expected_boxes,
        ", got ",
        nms_boxid_c.sizes()[0]);
    auto equal = (nms_boxid_c.sizes()[0] == num_expected_boxes);
    EXPECT_EQ(equal, true);
  }
}

TEST_F(LazyDynamicShapesTest, NmsSmall) {
  torch::manual_seed(0);

  auto num_boxes_cur = 8;
  auto num_boxes_var = 2;
  torch::Tensor scores_cur = torch::rand({num_boxes_cur});
  torch::Tensor boxes_cur = torch::rand({num_boxes_cur, 4}) * 256;

  while (num_boxes_cur < 13) {
    PRINT_TENSOR_WITH_DATA(boxes_cur);
    PRINT_TENSOR_WITH_DATA(scores_cur);

    auto num_expected_boxes{num_boxes_cur};

    if (num_expected_boxes) {
      torch::Tensor hscores = scores_cur.to(torch::kHPU);

      // Ensure x1 > x0 and y1 > y0
      auto tlist = boxes_cur.split(2, 1);
      tlist[1] = tlist[1] + tlist[0];
      auto valid_boxes = torch::cat({tlist[0], tlist[1]}, 1);
      // PRINT_TENSOR_WITH_DATA(valid_boxes);

      // Compute the iou scores
      // std::vector<std::vector<float>> iou_vec_2d;
      // iou_vec_2d.reserve(num_boxes_cur-1);
      // compute_iou(valid_boxes, iou_vec_2d);

      torch::Tensor hboxes = valid_boxes.to(torch::kHPU);

      auto nms_boxid = torchvision_nms_hpu_wrap(hboxes, hscores, 1.0);
      auto nms_boxid_c = nms_boxid.to(torch::kCPU);
      TORCH_CHECK(
          nms_boxid_c.dim() == 1,
          "Expecting a 1D tensor, got ",
          boxes_cur.dim(),
          "D tensor");
      // PRINT_TENSOR_WITH_DATA(scores_cur);
      // PRINT_TENSOR_WITH_DATA(nms_boxid_c);
      PT_TEST_DEBUG(
          "num_expected_boxes=",
          num_expected_boxes,
          ", got ",
          nms_boxid_c.sizes()[0]);
      auto equal = (nms_boxid_c.sizes()[0] == num_expected_boxes);
      EXPECT_EQ(equal, true);
    }

    // Generate boxes of random sizes
    torch::Tensor boxes_new = torch::rand({num_boxes_var, 4}) * 256;
    torch::Tensor scores_new = torch::rand({num_boxes_var});
    boxes_cur = torch::cat({boxes_cur, boxes_new}, 0);
    scores_cur = torch::cat({scores_cur, scores_new}, 0);
    num_boxes_cur += num_boxes_var;
  }
}

TEST_F(LazyDynamicShapesTest, BatchedNmsSmall) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  torch::manual_seed(0);
  float score_th = 0.2;

  auto num_boxes_cur = 10;
  auto num_boxes_var = 2;
  torch::Tensor scores_cur = torch::rand({num_boxes_cur});
  torch::Tensor boxes_cur = torch::rand({num_boxes_cur, 4}) * 256;
  torch::Tensor classes_cur =
      torch::randint(0, 1, {num_boxes_cur}, torch::kInt32);
  auto ref = torch::tensor({7, 1, 5, 0, 6, 8, 4}).to(torch::kLong);

  while (num_boxes_cur < 15) {
    PRINT_TENSOR_WITH_DATA(boxes_cur);
    PRINT_TENSOR_WITH_DATA(scores_cur);
    PRINT_TENSOR_WITH_DATA(classes_cur);

    auto num_expected_boxes{0};
    for (size_t i = 0; i < num_boxes_cur; i++) {
      float score = scores_cur[i].item<float>();
      if (score >= score_th) {
        num_expected_boxes++;
      }
    }

    if (num_expected_boxes) {
      torch::Tensor hscores = scores_cur.to(torch::kHPU);

      // Ensure x1 > x0 and y1 > y0
      auto tlist = boxes_cur.split(2, 1);
      tlist[1] = tlist[1] + tlist[0];
      auto valid_boxes = torch::cat({tlist[0], tlist[1]}, 1);
      // PRINT_TENSOR_WITH_DATA(valid_boxes);

      // Compute the iou scores
      // std::vector<std::vector<float>> iou_vec_2d;
      // iou_vec_2d.reserve(num_boxes_cur-1);
      // compute_iou(valid_boxes, iou_vec_2d);

      torch::Tensor hboxes = valid_boxes.to(torch::kHPU);

      torch::Tensor hclasses = classes_cur.to(torch::kHPU);

      auto nms_boxid =
          batched_nms_hpu_lazy(hboxes, hscores, hclasses, score_th);
      auto nms_boxid_c = nms_boxid.to(torch::kCPU);
      TORCH_CHECK(
          nms_boxid_c.dim() == 1,
          "Expecting a 1D tensor, got ",
          boxes_cur.dim(),
          "D tensor");
      // PRINT_TENSOR_WITH_DATA(scores_cur);
      // PRINT_TENSOR_WITH_DATA(nms_boxid_c);
      PT_TEST_DEBUG(
          "With score threshold=",
          score_th,
          ", num_expected_boxes=",
          num_expected_boxes,
          ", got ",
          nms_boxid_c.sizes()[0]);
      bool equal = ref.allclose(nms_boxid_c, 0, 0);
      EXPECT_EQ(equal, true);
    }

    // Generate boxes of random sizes
    torch::Tensor boxes_new = torch::rand({num_boxes_var, 4}) * 256;
    torch::Tensor scores_new = torch::rand({num_boxes_var});
    torch::Tensor classes_new =
        torch::randint(0, 1, {num_boxes_var}, torch::kInt32);
    boxes_cur = torch::cat({boxes_cur, boxes_new}, 0);
    scores_cur = torch::cat({scores_cur, scores_new}, 0);
    classes_cur = torch::cat({classes_cur, classes_new}, 0);
    if (num_boxes_cur == 10) {
      ref = torch::tensor({11, 10, 5, 0, 6, 8, 4, 3, 2}).to(torch::kLong);
    } else {
      ref = torch::tensor({11, 13, 10, 5, 0, 6, 3, 2, 12}).to(torch::kLong);
    }
    num_boxes_cur += num_boxes_var;
  }
  // while (score_th < 1.0) {
  // score_th += score_inc;
  //}
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyDynamicShapesTest, ArgmaxTest) {
  int N = 1;
  int C = 4;
  int H = 4;
  std::vector<int> in_sizes{6, 12, 20, 10};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    torch::Tensor A = torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor out_hpu = torch::argmax(hA, 2);
    torch::Tensor out_cpu = torch::argmax(A, 2);
    auto out = out_hpu.to(torch::kCPU);
    EXPECT_TRUE(allclose(out, out_cpu, 0, 0));
  }
}

TEST_F(LazyDynamicShapesTest, MaskRcnnGatherNdMxNetTest) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  int64_t dim = 0;
  int H = 4;
  std::vector<int> in_sizes{8000, 9000, 10000};
  std::vector<int> end_sizes{1000, 2000, 2000};
  std::vector<int> step_sizes{1, 2, 2};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    int index_size = W;
    torch::Tensor A = torch::randn({W, H});
    torch::Tensor B = torch::randn({W});

    torch::Tensor index =
        torch::randint(0, (W - 1), {index_size}, torch::dtype(torch::kInt64));

    // Make list
    c10::List<c10::optional<at::Tensor>> indices_cpu;
    c10::List<c10::optional<at::Tensor>> indices_list{};
    at::Tensor temp = torch::slice(index, 0, 0, end_sizes[i], step_sizes[i]);
    indices_cpu.push_back(c10::make_optional(temp));
    temp =
        torch::slice(index.to(torch::kHPU), 0, 0, end_sizes[i], step_sizes[i]);
    indices_list.push_back(c10::make_optional(temp));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hOut = torch::index(hA, indices_list);
    torch::Tensor out = torch::index(A, indices_cpu);
    torch::Tensor hOut1 = torch::index(hB, indices_list);
    torch::Tensor out1 = torch::index(B, indices_cpu);
    HbLazyTensor::StepMarker({});
    EXPECT_EQ(allclose(hOut.to(torch::kCPU), out, 0.001, 0.001), true);
  }
}

TEST_F(LazyDynamicShapesTest, MaskRcnnGatherNdMxNetTest1) {
  // iteration 1
  torch::Tensor input_cpu = torch::arange(4).reshape({2, 2});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu{torch::tensor({1}), torch::tensor({0, 1})};
  c10::List<c10::optional<at::Tensor>> indices_cpu{};
  indices_cpu.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_cpu.push_back(c10::make_optional(t));
  }
  c10::List<c10::optional<at::Tensor>> indices_list{};
  indices_list.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_list.push_back(c10::make_optional(t.to(torch::kHPU)));
  }
  auto out_cpu = at::index(input_cpu, indices_cpu);
  auto out_hpu = at::index(input_hpu, indices_list);

  HbLazyTensor::StepMarker({});

  // Iteration #2
  torch::Tensor input_cpu1 = torch::arange(4).reshape({2, 2});
  torch::Tensor input_hpu1 = input_cpu1.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu1{
      torch::tensor({1, 0}), torch::tensor({0})};
  c10::List<c10::optional<at::Tensor>> indices_cpu1{};
  indices_cpu1.reserve(vec_cpu1.size());
  for (auto t : vec_cpu1) {
    indices_cpu1.push_back(c10::make_optional(t));
  }
  c10::List<c10::optional<at::Tensor>> indices_list1{};
  indices_list1.reserve(vec_cpu1.size());
  for (auto t : vec_cpu1) {
    indices_list1.push_back(c10::make_optional(t.to(torch::kHPU)));
  }
  auto out_cpu1 = at::index(input_cpu1, indices_cpu1);
  auto out_hpu1 = at::index(input_hpu1, indices_list1);

  // Comparison
  EXPECT_EQ(allclose(out_hpu1.to(torch::kCPU), out_cpu1, 0.001, 0.001), true);
}

void runTopkDynamicTest(
    std::vector<int> K_values,
    std::vector<int> changing_dim_values,
    int dim) {
  int N = 1;
  int C = 2;
  int H = 2;
  int W = 20;
  c10::ScalarType dtype{torch::kInt32};
  std::vector<int64_t> dimensions = {N, C, H, W};
  for (int i = 0; i < K_values.size(); i++) {
    int change_dim_value = changing_dim_values[i];
    int k = K_values[i];
    dimensions[dim] = change_dim_value;
    PT_TEST_DEBUG("\nPTI_DBG :: TopKTest TEST ", i, "  --------\n");

    torch::Tensor input_cpu =
        torch::randn(dimensions, torch::requires_grad(false));
    // PRINT_TENSOR_WITH_DATA(input_cpu);

    torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

    std::tuple<at::Tensor, at::Tensor> out_hpu = torch::topk(input_hpu, k);
    std::tuple<at::Tensor, at::Tensor> out_cpu = torch::topk(input_cpu, k);

    auto hpu_value0 = std::get<0>(out_hpu);
    auto hpu_value1 = std::get<1>(out_hpu);

    auto cpu_value0 = std::get<0>(out_cpu);

    HbLazyTensor::StepMarker({});
    auto out0 = hpu_value0.to(torch::kCPU);
    auto out1 = hpu_value1.to(torch::kCPU);
    PT_TEST_DEBUG(
        "PTI_DBG :: input_cpu.shape : ",
        input_cpu.sizes(),
        " input_cpu.strides : ",
        input_cpu.strides());
    PT_TEST_DEBUG("PTI_DBG :: dimensions : ", dimensions, ", k : ", k);
    PT_TEST_DEBUG(
        "PTI_DBG :: output.shape : ",
        out0.sizes(),
        " output.strides : ",
        out0.strides());

    EXPECT_EQ(allclose(cpu_value0, out0, 0, 0), true);
  }
}

TEST_F(LazyDynamicShapesTest, TopKTest1) {
  // Changing K values
  runTopkDynamicTest({5, 15, 25, 20, 6, 8}, {30, 30, 30, 30, 30, 30}, 3);
  // Changing W values
  runTopkDynamicTest({5, 5, 5, 5, 5, 5}, {20, 33, 40, 35, 25, 28}, 3);
  // Changing K and W values
  runTopkDynamicTest({5, 15, 25, 20, 6, 8}, {20, 33, 40, 35, 25, 28}, 3);
}

void runSortDynamicTest(std::vector<int> changing_dim_values, int dim) {
  int N = 1;
  int C = 2;
  int H = 2;
  int W = 20;
  c10::ScalarType dtype{torch::kInt32};
  std::vector<int64_t> dimensions = {N, C, H, W};
  for (int i = 0; i < changing_dim_values.size(); i++) {
    int change_dim_value = changing_dim_values[i];
    dimensions[dim] = change_dim_value;
    PT_TEST_DEBUG("\nPTI_DBG :: SortTest TEST ", i, "  --------\n");

    torch::Tensor input_cpu =
        torch::randn(dimensions, torch::requires_grad(false));
    // PRINT_TENSOR_WITH_DATA(input_cpu);

    torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

    std::tuple<at::Tensor, at::Tensor> out_hpu = torch::sort(input_hpu, dim);
    std::tuple<at::Tensor, at::Tensor> out_cpu = torch::sort(input_cpu, dim);

    auto hpu_value0 = std::get<0>(out_hpu);
    auto hpu_value1 = std::get<1>(out_hpu);

    auto cpu_value0 = std::get<0>(out_cpu);

    HbLazyTensor::StepMarker({});
    auto out0 = hpu_value0.to(torch::kCPU);
    auto out1 = hpu_value1.to(torch::kCPU);
    PT_TEST_DEBUG(
        "PTI_DBG :: input_cpu.shape : ",
        input_cpu.sizes(),
        " input_cpu.strides : ",
        input_cpu.strides());
    PT_TEST_DEBUG("PTI_DBG :: dimensions : ", dimensions);
    PT_TEST_DEBUG(
        "PTI_DBG :: output.shape : ",
        out0.sizes(),
        " output.strides : ",
        out0.strides());

    EXPECT_EQ(allclose(cpu_value0, out0, 0, 0), true);
  }
}

TEST_F(LazyDynamicShapesTest, SortTest1) {
  // Changing W values
  runSortDynamicTest({20, 33, 40, 35, 25, 28}, 3);
}

void runTopkOutDynamicTest(
    std::vector<int> K_values,
    std::vector<int> changing_dim_values,
    int dim) {
  std::vector<int64_t> dimensions = {1, 2, 2, 20};
  for (int i = 0; i < K_values.size(); i++) {
    int change_dim_value = changing_dim_values[i];
    int k = K_values[i];
    dimensions[dim] = change_dim_value;
    PT_TEST_DEBUG("\nPTI_DBG :: TopKOutTest TEST ", i, "  --------\n");

    torch::Tensor input_cpu =
        torch::randn(dimensions, torch::requires_grad(false));
    torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

    auto cpu_values = torch::empty(0);
    auto hpu_values = cpu_values.to(torch::kHPU);

    auto cpu_indices = torch::empty(0).to(torch::kLong);
    auto hpu_indices = cpu_indices.to(torch::kHPU);

    torch::topk_out(hpu_values, hpu_indices, input_hpu, k);
    torch::topk_out(cpu_values, cpu_indices, input_cpu, k);

    HbLazyTensor::StepMarker({});
    auto out0 = hpu_values.to(torch::kCPU);
    auto out1 = hpu_indices.to(torch::kCPU);
    PT_TEST_DEBUG(
        "PTI_DBG :: input_cpu.shape : ",
        input_cpu.sizes(),
        " input_cpu.strides : ",
        input_cpu.strides());
    PT_TEST_DEBUG("PTI_DBG :: dimensions : ", dimensions, ", k : ", k);
    PT_TEST_DEBUG(
        "PTI_DBG :: output.shape : ",
        out0.sizes(),
        " output.strides : ",
        out0.strides());

    EXPECT_EQ(allclose(cpu_values, out0, 0, 0), true);
  }
}

TEST_F(LazyDynamicShapesTest, TopKOutTest) {
  // Changing K values
  runTopkOutDynamicTest({5, 15, 25, 20, 6, 8}, {30, 30, 30, 30, 30, 30}, 3);
  // Changing W values
  runTopkOutDynamicTest({5, 5, 5, 5, 5, 5}, {20, 33, 40, 35, 25, 28}, 3);
  // Changing K and W values
  runTopkOutDynamicTest({5, 15, 25, 20, 6, 8}, {20, 33, 40, 35, 25, 28}, 3);
}

void runSortOutDynamicTest(std::vector<int> changing_dim_values, int dim) {
  std::vector<int64_t> dimensions = {1, 2, 2, 20};
  for (int i = 0; i < changing_dim_values.size(); i++) {
    int change_dim_value = changing_dim_values[i];
    dimensions[dim] = change_dim_value;
    PT_TEST_DEBUG("\nPTI_DBG :: TopKOutTest TEST ", i, "  --------\n");

    torch::Tensor input_cpu =
        torch::randn(dimensions, torch::requires_grad(false));
    torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

    auto cpu_values = torch::empty(0);
    auto hpu_values = cpu_values.to(torch::kHPU);

    auto cpu_indices = torch::empty(0).to(torch::kLong);
    auto hpu_indices = cpu_indices.to(torch::kHPU);

    torch::sort_out(hpu_values, hpu_indices, input_hpu, dim);
    torch::sort_out(cpu_values, cpu_indices, input_cpu, dim);

    HbLazyTensor::StepMarker({});
    auto out0 = hpu_values.to(torch::kCPU);
    auto out1 = hpu_indices.to(torch::kCPU);
    PT_TEST_DEBUG(
        "PTI_DBG :: input_cpu.shape : ",
        input_cpu.sizes(),
        " input_cpu.strides : ",
        input_cpu.strides());
    PT_TEST_DEBUG("PTI_DBG :: dimensions : ", dimensions);
    PT_TEST_DEBUG(
        "PTI_DBG :: output.shape : ",
        out0.sizes(),
        " output.strides : ",
        out0.strides());

    EXPECT_EQ(allclose(cpu_values, out0, 0, 0), true);
  }
}

void runConstantPadDynamicTest(std::vector<int> Pad, int val) {
  c10::ScalarType dtype{torch::kInt32};
  for (int i = 0; i < 4; i++) {
    torch::Tensor input_cpu =
        torch::randn({1 + i, 6}, torch::requires_grad(false));

    torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
    auto result = torch::constant_pad_nd(input_cpu, {2, 3}, 5);
    auto result_hpu = torch::constant_pad_nd(input_hpu, {2, 3}, 5);

    HbLazyTensor::StepMarker({});
    auto out = result_hpu.to(torch::kCPU);

    EXPECT_EQ(allclose(result, out, 0, 0), true);
  }
}
TEST_F(LazyDynamicShapesTest, ConstantPad) {
  runConstantPadDynamicTest({2}, 9);
}

TEST_F(LazyDynamicShapesTest, SortOutTest) {
  // Changing W values
  runSortOutDynamicTest({20, 33, 40, 35, 25, 28}, 3);
  // Changing W values
  runSortOutDynamicTest({20, 33, 40, 35, 25, 28}, 2);
}

TEST_F(LazyDynamicShapesTest, DS_RoiAlignFwdTest) {
  auto roi_align_test = [](int num_boxes, std::vector<int64_t> input_shape) {
    auto images = torch::randn(input_shape).to(torch::kHPU);
    auto boxes = torch::randn({num_boxes, 4}) * 64;
    // ensure x2 > x1 and y2 > y1
    auto tlist = boxes.split(2, 1);
    tlist[1] = tlist[1] + tlist[0];
    auto new_boxes = torch::cat({tlist[0], tlist[1]}, 1).to(torch::kHPU);
    auto num_rois =
        torch::randint(0, 2, {num_boxes}, torch::kInt).to(torch::kHPU);
    auto output = roi_align_fwd_hpu_lazy(
        images, new_boxes, num_rois, 7, 7, 0, 2, 0.25, true);
    output.to(torch::kCPU);
  };
  roi_align_test(6, {2, 3, 25, 25});
  roi_align_test(10, {2, 3, 35, 35});
  roi_align_test(12, {2, 3, 50, 50});
}

TEST_F(LazyDynamicShapesTest, ConvSliceReluChLastTest) {
  int kH = 3;
  int kW = 3;
  const int C = 1;
  const int N = 4;
  int H = 16;

  std::vector<int> in_sizes{16, 32, 64};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    int W = in_sizes[i];
    torch::Tensor weight_tensor =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false)).contiguous();
    torch::Tensor h_weight_tensor = weight_tensor.to(torch::kHPU);
    torch::Tensor in_tensor =
        torch::randn({N, C, H, W}, torch::requires_grad(false))
            .contiguous(c10::MemoryFormat::ChannelsLast);
    torch::Tensor h_in_tensor = in_tensor.to(torch::kHPU);
    torch::Tensor h_weight_tensor_hwck = h_weight_tensor;

    // conv2d
    torch::Tensor h_out_conv = torch::conv2d(
        h_in_tensor, h_weight_tensor_hwck, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor out_conv = torch::conv2d(
        in_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);

    // Slice
    auto h_slice_out = torch::slice(h_out_conv, 3, 1, 4, 1);
    auto slice_out = torch::slice(out_conv, 3, 1, 4, 1);

    // Relu
    torch::Tensor h_relu_out = torch::relu(h_slice_out);
    torch::Tensor relu_out = torch::relu(slice_out);

    torch::Tensor out_hpu = h_relu_out.to(torch::kCPU);
    EXPECT_EQ(allclose(out_hpu, relu_out, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
}

TEST_F(LazyDynamicShapesTest, RandpermOutTest) {
  std::vector<int> in_sizes{8, 10, 15};
  for (int i = 0; i < in_sizes.size(); i++) {
    int n = in_sizes[i];
    c10::optional<at::ScalarType> dtype = c10::ScalarType::Int;
    c10::optional<at::Device> hb_device = at::DeviceType::HPU;
    at::TensorOptions hb_options =
        at::TensorOptions().dtype(dtype).device(hb_device);
    torch::manual_seed(0);
    auto lazy = torch::randperm(n, hb_options);
    auto lazy_cpu = lazy.to(torch::kCPU);
  }
}

TEST_F(LazyDynamicShapesTest, ScatterTest) {
  auto scatter_test = [](std::vector<int64_t> in_shape1,
                         std::vector<int64_t> in_shape2) {
    torch::Tensor a = torch::randn(in_shape1, torch::requires_grad(false));
    torch::Tensor h_a = a.to(torch::kHPU);
    int64_t dim = 0;
    auto index =
        torch::randint(0, in_shape1[0], in_shape2, torch::dtype(torch::kInt64));
    auto h_index = index.to(torch::kHPU);
    auto value = 2;

    h_a.add_(1);
    h_a.scatter_(dim, h_index, value);
    h_a.add_(1);
    auto h_cout = h_a.to(torch::kCPU);
    a.add_(1);
    a.scatter_(dim, index, value);
    a.add_(1);

    EXPECT_EQ(allclose(h_cout, a), true);
  };
  scatter_test({500}, {20});
  scatter_test({500}, {25});
  scatter_test({500}, {30});
}

TEST_F(LazyDynamicShapesTest, MaxTest) {
  auto max_test = [](std::vector<int64_t> input_shape) {
    auto input = torch::randn(input_shape);
    auto input_h = input.to(torch::kHPU);
    auto out = input.max();
    auto out_h = input_h.max();
    EXPECT_EQ(allclose(out_h.to(torch::kCPU), out, 0.001, 0.001), true);
  };
  max_test({2, 10});
  max_test({2, 12});
  max_test({2, 15});
}

TEST_F(LazyDynamicShapesTest, DS_PadTest_HT) {
  auto pad_test = [](std::vector<int64_t> pad_sizes,
                     std::vector<int64_t> input_shape) {
    torch::Tensor tensor = torch::randn(input_shape).to(torch::kInt);
    torch::Tensor tensorHabana = tensor.to(torch::kHPU);

    namespace F = torch::nn::functional;
    auto outHabana = F::pad(
        tensorHabana,
        F::PadFuncOptions(pad_sizes).mode(torch::kConstant).value(0.0));
    auto out = F::pad(
        tensor, F::PadFuncOptions(pad_sizes).mode(torch::kConstant).value(0.0));
    bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
    EXPECT_EQ(equal, true);
  };
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  pad_test({0, 14, 0, 0}, {3, 80, 122});
  pad_test({0, 0, 0, 22}, {3, 87, 80});
  pad_test({0, 28, 0, 0}, {3, 80, 106});
  pad_test({0, 0, 0, 20}, {3, 119, 80});
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyDynamicShapesTest, DS_PadTest) {
  SET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_PAD_HOST_TENSOR, false, 1);
  auto pad_test = [](std::vector<int64_t> pad_sizes,
                     std::vector<int64_t> input_shape) {
    torch::Tensor tensor = torch::randn(input_shape).to(torch::kInt);
    torch::Tensor tensorHabana = tensor.to(torch::kHPU);

    namespace F = torch::nn::functional;
    auto outHabana = F::pad(
        tensorHabana,
        F::PadFuncOptions(pad_sizes).mode(torch::kConstant).value(0.0));
    auto out = F::pad(
        tensor, F::PadFuncOptions(pad_sizes).mode(torch::kConstant).value(0.0));
    bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
    EXPECT_EQ(equal, true);
  };
  pad_test({0, 14, 0, 0}, {3, 80, 122});
  pad_test({0, 0, 0, 22}, {3, 87, 80});
  pad_test({0, 28, 0, 0}, {3, 80, 106});
  pad_test({0, 0, 0, 20}, {3, 119, 80});
  UNSET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_PAD_HOST_TENSOR);
}

TEST_F(LazyDynamicShapesTest, DISABLED_DS_GridSamplerTest) {
  auto grid_sampler_test = [](std::vector<int64_t> pad_sizes,
                              std::vector<int64_t> input_shape) {
    torch::Tensor tensor = torch::randn(pad_sizes).to(torch::kFloat);
    torch::Tensor tensorHabana = tensor.to(torch::kHPU);
    torch::Tensor grid_tensor = torch::randn(input_shape).to(torch::kFloat);
    torch::Tensor gridHabana = grid_tensor.to(torch::kHPU);
    namespace F = torch::nn::functional;
    auto outHabana = F::grid_sample(
        tensorHabana,
        gridHabana,
        F::GridSampleFuncOptions()
            .mode(torch::kBilinear)
            .padding_mode(torch::kZeros)
            .align_corners(false));
    auto out = F::grid_sample(
        tensor,
        grid_tensor,
        F::GridSampleFuncOptions()
            .mode(torch::kBilinear)
            .padding_mode(torch::kZeros)
            .align_corners(false));
    bool equal = out.allclose(outHabana.to(torch::kCPU), 0.000001, 0.000001);
    EXPECT_EQ(equal, true);
  };
  grid_sampler_test({2, 3, 4, 4}, {2, 10, 10, 2});
  grid_sampler_test({1, 1, 2, 2}, {1, 3, 3, 2});
  grid_sampler_test({1, 1, 2, 2}, {1, 4, 4, 2});
}

void runIndexPutDynamicTestBool(int N, bool acc) {
  std::vector<int64_t> dimensions = {N};

  torch::Tensor input_cpu =
      torch::randn(dimensions, torch::requires_grad(false));

  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  torch::Tensor mask_cpu = input_cpu > 0;

  torch::Tensor mask_hpu = mask_cpu.to(torch::kHPU);

  torch::Tensor values_cpu = torch::tensor(0.0);
  auto values_hpu = values_cpu.to(torch::kHPU);

  torch::Tensor out_hpu =
      torch::index_put(input_hpu, {mask_hpu}, values_hpu, acc);
  torch::Tensor out_cpu =
      torch::index_put(input_cpu, {mask_cpu}, values_cpu, acc);

  auto hpu_out_to_cpu = out_hpu.to(torch::kCPU);

  EXPECT_EQ(allclose(out_cpu, out_hpu, 0, 0), true);
}

void runIndexPutDynamicTestInt(int N, int mask_size, bool acc) {
  std::vector<int64_t> dimensions = {N};

  torch::Tensor input_cpu =
      torch::randn(dimensions, torch::requires_grad(false));

  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  torch::Tensor mask_cpu = torch::randint(1, N - 1, {mask_size}, torch::kInt64);

  torch::Tensor mask_hpu = mask_cpu.to(torch::kHPU);

  torch::Tensor values_cpu = torch::tensor(3.0);
  auto values_hpu = values_cpu.to(torch::kHPU);

  torch::Tensor out_hpu =
      torch::index_put(input_hpu, {mask_hpu}, values_hpu, acc);
  torch::Tensor out_cpu =
      torch::index_put(input_cpu, {mask_cpu}, values_cpu, acc);

  auto hpu_out_to_cpu = out_hpu.to(torch::kCPU);

  EXPECT_EQ(allclose(out_cpu, out_hpu, 0, 0), true);
}

TEST_F(LazyDynamicShapesTest, IndexPutAccBoolTest) {
  runIndexPutDynamicTestBool(1645, true);
  runIndexPutDynamicTestBool(1655, true);
  runIndexPutDynamicTestBool(1665, true);
}

TEST_F(LazyDynamicShapesTest, IndexPutAccIntTest) {
  runIndexPutDynamicTestInt(16145, 20, true);
  runIndexPutDynamicTestInt(19155, 40, true);
  runIndexPutDynamicTestInt(22165, 60, true);
}

TEST_F(LazyDynamicShapesTest, IndexPutNonAccBoolTest) {
  runIndexPutDynamicTestBool(1645, false);
  runIndexPutDynamicTestBool(1955, false);
  runIndexPutDynamicTestBool(2265, false);
}

TEST_F(LazyDynamicShapesTest, IndexPutNonAccIntTest) {
  runIndexPutDynamicTestInt(16145, 20, false);
  runIndexPutDynamicTestInt(19155, 40, false);
  runIndexPutDynamicTestInt(22165, 60, false);
}

void runIndexPutDynamicTestBoolVect(
    std::vector<int64_t> input_dim,
    std::vector<int64_t> mask_dim,
    std::vector<int64_t> values_dim,
    bool acc) {
  std::vector<int64_t> dimensions = input_dim;

  torch::Tensor input_cpu =
      torch::randn(dimensions, torch::requires_grad(false));

  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  torch::Tensor mask_cpu;

  if (mask_dim.empty()) {
    mask_cpu = torch::zeros(input_dim, torch::kBool);
  } else {
    mask_cpu = torch::zeros(mask_dim, torch::kBool);
  }

  for (int i = 0; i < values_dim[0]; i++) {
    mask_cpu[0][i] = true;
  }

  torch::Tensor mask_hpu = mask_cpu.to(torch::kHPU);

  torch::Tensor values_cpu =
      torch::randn(values_dim, torch::requires_grad(false));
  auto values_hpu = values_cpu.to(torch::kHPU);

  torch::Tensor out_hpu =
      torch::index_put(input_hpu, {mask_hpu}, values_hpu, acc);

  torch::Tensor out_cpu =
      torch::index_put(input_cpu, {mask_cpu}, values_cpu, acc);

  auto hpu_out_to_cpu = out_hpu.to(torch::kCPU);

  EXPECT_EQ(allclose(out_cpu, out_hpu, 0, 0), true);
}

TEST_F(LazyDynamicShapesTest, IndexPutAccBoolTestNC) {
  runIndexPutDynamicTestBoolVect(
      {2, 1645},
      {},
      {
          512,
      },
      true);
  runIndexPutDynamicTestBoolVect(
      {4, 1955},
      {},
      {
          1024,
      },
      true);
  runIndexPutDynamicTestBoolVect({8, 2265}, {}, {2048}, true);
}

TEST_F(LazyDynamicShapesTest, IndexPutNonAccBoolTestNC) {
  runIndexPutDynamicTestBoolVect(
      {2, 1645},
      {},
      {
          512,
      },
      false);
  runIndexPutDynamicTestBoolVect(
      {4, 1955},
      {},
      {
          1024,
      },
      false);
  runIndexPutDynamicTestBoolVect({8, 2265}, {}, {2048}, false);
}

TEST_F(LazyDynamicShapesTest, IndexPutAccBoolTestNCH) {
  /* These are MaskRCNN original configurations - Reducing the dimensionality
 while testing to reduce CI time Keeping in comments to try out in future if
 necessary. runIndexPutDynamicTestBoolVect({2, 161145, 4}, {2, 161145}, {47, 4},
 true/false); runIndexPutDynamicTestBoolVect({4, 191155, 8}, {4, 191155}, {94,
 8}, true/false); runIndexPutDynamicTestBoolVect({8, 221165, 16}, {8, 221165},
 {188, 16}, true/false);
 */
  auto indexPutTest = [](bool acc) {
    runIndexPutDynamicTestBoolVect({2, 16, 4}, {2, 16}, {4, 4}, acc);
    runIndexPutDynamicTestBoolVect({4, 19, 8}, {4, 19}, {9, 8}, acc);
    runIndexPutDynamicTestBoolVect({8, 22, 16}, {8, 22}, {18, 16}, acc);
  };
  indexPutTest(true);
  indexPutTest(false);
}

void runIndexPutDynamicTestIntVect(
    std::vector<int64_t> input_dim,
    std::vector<int64_t> mask_dim,
    std::vector<int64_t> values_dim,
    bool acc) {
  std::vector<int64_t> dimensions = input_dim;

  torch::Tensor input_cpu =
      torch::randn(dimensions, torch::requires_grad(false));

  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  torch::Tensor mask_cpu1 =
      torch::randint(0, input_dim[0], mask_dim, torch::kInt64);

  torch::Tensor mask_hpu1 = mask_cpu1.to(torch::kHPU);

  torch::Tensor mask_cpu2 =
      torch::randint(0, input_dim[0], mask_dim, torch::kInt64);

  torch::Tensor mask_hpu2 = mask_cpu2.to(torch::kHPU);

  // torch::Tensor values_cpu = torch::tensor(0.0);
  torch::Tensor values_cpu =
      torch::randn(values_dim, torch::requires_grad(false));
  auto values_hpu = values_cpu.to(torch::kHPU);

  torch::Tensor out_hpu;
  torch::Tensor out_cpu;

  if (acc) {
    out_hpu =
        torch::index_put(input_hpu, {mask_hpu1, mask_hpu2}, values_hpu, acc);
    out_cpu =
        torch::index_put(input_cpu, {mask_cpu1, mask_cpu2}, values_cpu, acc);
  } else {
    out_hpu = torch::index_put(input_hpu, {mask_hpu1}, values_hpu, acc);
    out_cpu = torch::index_put(input_cpu, {mask_cpu1}, values_cpu, acc);
  }
}

void repeatInlvTest(
    at::Tensor A,
    std::vector<int64_t> rpt_vals,
    int64_t dim = -1) {
  auto rpt = torch::tensor(rpt_vals);
  auto hrpt = rpt.to(torch::kHPU);
  auto hA = A.to(torch::kHPU);
  if (dim != -1) {
    auto hOut = hA.repeat_interleave(hrpt, dim);
    auto Out = A.repeat_interleave(rpt, dim);
    EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out));
  } else {
    auto hOut = hA.repeat_interleave(hrpt);
    auto Out = A.repeat_interleave(rpt);
    EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out));
  }
}

TEST_F(LazyDynamicShapesTest, RepeatInlv1) {
  repeatInlvTest(torch::tensor({4, 5}), {10, 7});
  repeatInlvTest(torch::tensor({4, 5}), {15, 8});
  repeatInlvTest(torch::tensor({4, 5}), {20, 10});
}

TEST_F(LazyDynamicShapesTest, RepeatInlv2) {
  repeatInlvTest(torch::randn({4, 5}), {2});
  repeatInlvTest(torch::randn({4, 5}), {3});
  repeatInlvTest(torch::randn({4, 5}), {4});
}

TEST_F(LazyDynamicShapesTest, RepeatInlv3) {
  repeatInlvTest(torch::randn({4, 5}), {2, 1, 1, 1, 1}, 1);
  repeatInlvTest(torch::randn({4, 5}), {2, 1, 2, 1, 2}, 1);
  repeatInlvTest(torch::randn({4, 5}), {2, 2, 2, 2, 2}, 1);
}

TEST_F(LazyDynamicShapesTest, EvictRecipeSingleOpRelu) {
  std::vector<int> in_sizes{6, 8, 10, 20, 50};
  int rounds{2};

  const char* recipe_cache_path = GET_ENV_FLAG_NEW(PT_RECIPE_CACHE_PATH);
  uint32_t initial_host_mem_threshold =
      GET_ENV_FLAG_NEW(PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT);
  habana::RecipeCacheLRU::SetHostMemoryThreshold(100);
  while (rounds--) {
    for (int i = 0; i < in_sizes.size(); i++) {
      int B = in_sizes[i];
      torch::Tensor input_cpu =
          torch::randn({3, B, 4}, torch::requires_grad(false));
      torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
      torch::Tensor out_hpu = torch::relu(input_hpu);
      HbLazyTensor::StepMarker({});
    }
  }

  auto actual_recipe_count = habana::RecipeCacheLRU::get_cache().get_length();
  habana::RecipeCacheLRU::get_cache().clear();

  UNSET_ENV_FLAG_NEW(PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT);
  habana::RecipeCacheLRU::SetHostMemoryThreshold(1);
  rounds = 2;
  while (rounds--) {
    for (int i = 0; i < in_sizes.size(); i++) {
      int B = in_sizes[i];
      torch::Tensor input_cpu =
          torch::randn({3, B, 4}, torch::requires_grad(false));
      torch::Tensor out_cpu = torch::relu(input_cpu);

      torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
      torch::Tensor out_hpu = torch::relu(input_hpu);
      torch::Tensor out_hpu_c = out_hpu.to(torch::kCPU);
      EXPECT_EQ(allclose(out_cpu, out_hpu_c, 0.01, 0.01), true);
    }
  }

  UNSET_ENV_FLAG_NEW(PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT);
  habana::RecipeCacheLRU::SetHostMemoryThreshold(initial_host_mem_threshold);

  auto current_recipe_count = habana::RecipeCacheLRU::get_cache().get_length();
  HABANA_ASSERT(
      (current_recipe_count < actual_recipe_count) ||
      (recipe_cache_path != nullptr));
}

TEST_F(LazyDynamicShapesTest, BatchNormFwdBwdDS) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  const int N = 16;
  int H = 16;
  at::Scalar inScalar = 2.0;
  std::vector<int> in_sizes{16, 64};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    int W = in_sizes[i];
    torch::Tensor in_tensor =
        torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor h_in_tensor = in_tensor.to(torch::kHPU);
    torch::Tensor gamma =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor beta =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    c10::optional<at::Tensor> mean;
    torch::Tensor var =
        torch::ones(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor h_gamma = gamma.to(torch::kHPU);
    torch::Tensor h_beta = beta.to(torch::kHPU);
    torch::Tensor h_var = var.to(torch::kHPU);
    float mom = 0.1;
    float eps = 1e-5;
    auto h_bn_outs = torch::native_batch_norm(
        h_in_tensor, h_gamma, h_beta, mean, h_var, true, mom, eps);
    auto bn_outs = torch::native_batch_norm(
        in_tensor, gamma, beta, mean, var, true, mom, eps);
    auto h_bn_out = std::get<0>(h_bn_outs);
    auto bn_out = std::get<0>(bn_outs);

    torch::Tensor out_hpu = h_bn_out.to(torch::kCPU);
    EXPECT_EQ(allclose(out_hpu, bn_out, 0.01, 0.01), true);

    auto grad_tensor = torch::randn({N, C, H, W}, torch::requires_grad(false));
    auto tHabanaGrad = grad_tensor.to(torch::kHPU);

    auto save_mean = torch::randn({C}, torch::requires_grad(false));
    auto tHabanaSaveMean = save_mean.to(torch::kHPU);

    auto save_ivar = torch::randn({C}, torch::requires_grad(false));
    auto tHabanaSaveIVar = save_ivar.to(torch::kHPU);

    auto results_cpu = torch::native_batch_norm_backward(
        grad_tensor,
        in_tensor,
        gamma,
        mean,
        var,
        save_mean,
        save_ivar,
        true,
        0.1,
        {true, true, true});
    at::Tensor result_cpu = std::get<0>(results_cpu);

    auto results = torch::native_batch_norm_backward(
        tHabanaGrad,
        h_in_tensor,
        h_gamma,
        mean,
        h_var,
        tHabanaSaveMean,
        tHabanaSaveIVar,
        true,
        0.1,
        {true, true, true});

    at::Tensor result_lazy = std::get<0>(results).to(torch::kCPU);
    EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
}

TEST_F(LazyDynamicShapesTest, stridedviewoutDynTest) {
  auto bucket = torch::randn({64});
  auto hbucket = bucket.to(torch::kHPU);

  auto gv1 = bucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 0);
  auto gv2 = bucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 16);
  auto gv3 = bucket.as_strided({32}, {1}, 32);

  auto g1 = torch::randn({2, 2, 2, 2});
  auto g2 = torch::randn({2, 2, 2, 2});
  auto g3 = torch::randn({32});

  gv1.mul_(g1);
  gv2.mul_(g2);
  gv3.mul_(g3);

  // hpu
  auto hgv1 = hbucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 0);
  auto hgv2 = hbucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 16);
  auto hgv3 = hbucket.as_strided({32}, {1}, 32);

  auto hg1 = g1.to(torch::kHPU);
  auto hg2 = g2.to(torch::kHPU);
  auto hg3 = g3.to(torch::kHPU);

  hgv1.mul_(hg1);
  hgv2.mul_(hg2);
  hgv3.mul_(hg3);

  HbLazyTensorViews::StepMarkerAllReduce({hbucket});

  EXPECT_EQ(allclose(hgv1.cpu(), gv1, 0.001, 0.001), true);
  EXPECT_EQ(allclose(hgv2.cpu(), gv2, 0.001, 0.001), true);
  EXPECT_EQ(allclose(hgv3.cpu(), gv3, 0.001, 0.001), true);

  // optimizer
  auto out = torch::mul(gv3, 0.1);
  auto hout = torch::mul(hgv3, 0.1);

  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(hout.cpu(), out, 0.001, 0.001), true);

  // cache hit case
  bucket = torch::randn({64});
  hbucket = bucket.to(torch::kHPU);
  gv1 = bucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 0);
  gv2 = bucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 16);
  gv3 = bucket.as_strided({32}, {1}, 32);

  g1 = torch::randn({2, 2, 2, 2});
  g2 = torch::randn({2, 2, 2, 2});
  g3 = torch::randn({32});

  gv1.mul_(g1);
  gv2.mul_(g2);
  gv3.mul_(g3);

  // hpu
  hgv1 = hbucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 0);
  hgv2 = hbucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 16);
  hgv3 = hbucket.as_strided({32}, {1}, 32);

  hg1 = g1.to(torch::kHPU);
  hg2 = g2.to(torch::kHPU);
  hg3 = g3.to(torch::kHPU);

  hgv1.mul_(hg1);
  hgv2.mul_(hg2);
  hgv3.mul_(hg3);

  HbLazyTensorViews::StepMarkerAllReduce({hbucket});

  EXPECT_EQ(allclose(hgv1.cpu(), gv1, 0.001, 0.001), true);
  EXPECT_EQ(allclose(hgv2.cpu(), gv2, 0.001, 0.001), true);
  EXPECT_EQ(allclose(hgv3.cpu(), gv3, 0.001, 0.001), true);

  // optimizer
  gv3.mul_(0.1);
  hgv3.mul_(0.1);

  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(hout.cpu(), out, 0.001, 0.001), true);
}

TEST_F(LazyDynamicShapesTest, ExponentialDynTest) {
  std::uniform_int_distribution<> dist(-127, 128);
  std::mt19937 m_mt_;
  double lambd = dist(m_mt_);
  std::vector<int> sizes = {3, 9, 12, 20, 60, 80};
  for (int i = 0; i < sizes.size(); i++) {
    torch::Tensor t0 =
        torch::randint(-127, 128, {3, sizes[i]}).to(torch::kFloat);
    torch::Tensor t0_h = t0.to("hpu");
    auto result0 = torch::exponential(t0_h, lambd);
    auto result1 = torch::exponential(t0_h, lambd);

    EXPECT_FALSE(torch::equal(result0.cpu(), result1.cpu()));
  }
}

TEST_F(LazyDynamicShapesTest, AsStridedH2DTest) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  if (false == GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED)) {
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED, true, 1);
  }

  std::vector<int64_t> in_sizes{6144, 24576, 98304};
  std::vector<std::vector<int64_t>> out_sizes{
      {2, 2, 32, 32}, {2, 2, 64, 64}, {2, 2, 128, 128}};
  std::vector<std::vector<int64_t>> strides{
      {3072, 1024, 32, 1}, {12288, 4096, 64, 1}, {49152, 16384, 128, 1}};
  std::vector<int64_t> offsets{1024, 4096, 16384};

  for (int i = 0; i < in_sizes.size(); i++) {
    auto in_s = in_sizes[i];
    c10::IntArrayRef out_s(out_sizes[i].data(), out_sizes[i].size());
    auto stride = strides[i];
    auto offset = offsets[i];

    PT_TEST_DEBUG("\n PTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({in_s});
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hOut = torch::as_strided(hA, out_s, stride, offset);
    torch::Tensor out = torch::as_strided(A, out_s, stride, offset);
    hOut = hOut.add_(1.0);
    out = out.add_(1.0);

    EXPECT_EQ(allclose(hOut.to(torch::kCPU), out, 0.001, 0.001), true);
  }

  SET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED, false, 1);
}

TEST_F(LazyDynamicShapesTest, AsStridedStrideRatioH2DTest) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  if (false == GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED)) {
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED, true, 1);
  }

  std::vector<std::vector<int64_t>> in_sizes{
      {2, 3, 32, 32}, {2, 3, 64, 64}, {2, 3, 128, 128}};
  std::vector<std::vector<int64_t>> out_sizes{
      {2, 2, 32, 32}, {2, 2, 64, 64}, {2, 2, 128, 128}};
  std::vector<std::vector<int64_t>> strides{
      {3072, 1024, 32, 1}, {12288, 4096, 64, 1}, {49152, 16384, 128, 1}};
  std::vector<int64_t> offsets{1024, 4096, 16384};

  for (int i = 0; i < in_sizes.size(); i++) {
    c10::IntArrayRef in_s(in_sizes[i].data(), in_sizes[i].size());
    c10::IntArrayRef out_s(out_sizes[i].data(), out_sizes[i].size());
    auto stride = strides[i];
    auto offset = offsets[i];

    PT_TEST_DEBUG("\n PTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn(in_s);
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hOut = torch::as_strided(hA, out_s, stride, offset);
    torch::Tensor out = torch::as_strided(A, out_s, stride, offset);
    hOut = hOut.add_(1.0);
    out = out.add_(1.0);

    EXPECT_EQ(allclose(hOut.to(torch::kCPU), out, 0.001, 0.001), true);
  }

  SET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED, false, 1);
}

// Reproducer for https://jira.habana-labs.com/browse/SW-117082
TEST_F(LazyDynamicShapesTest, AsStridedStrideRatioH2DTest_5D) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  auto enable_fast_sif = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE);
  auto run_hybrid_sif = GET_ENV_FLAG_NEW(PT_HPU_RUN_HYBRID_SIF);

  // Disable hybrid sif if enabled
  if (enable_fast_sif || run_hybrid_sif) {
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE, false, 1);
    SET_ENV_FLAG_NEW(PT_HPU_RUN_HYBRID_SIF, false, 1);
  }

  std::vector<std::vector<int64_t>> in_sizes{
      {1, 3, 32, 32, 32}, {1, 3, 128, 128, 128}};
  std::vector<std::vector<int64_t>> out_sizes{
      {1, 3, 32, 32, 32}, {1, 3, 128, 128, 128}};
  std::vector<std::vector<int64_t>> strides{
      {98304, 32768, 1024, 32, 1}, {6291456, 2097152, 16384, 128, 1}};
  std::vector<int64_t> offsets{0, 0};

  for (int i = 0; i < in_sizes.size(); i++) {
    c10::IntArrayRef in_s(in_sizes[i].data(), in_sizes[i].size());
    c10::IntArrayRef out_s(out_sizes[i].data(), out_sizes[i].size());
    auto stride = strides[i];
    auto offset = offsets[i];

    PT_TEST_DEBUG("\n PTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn(in_s);
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hOut = torch::as_strided(hA, out_s, stride, offset);
    torch::Tensor out = torch::as_strided(A, out_s, stride, offset);
    hOut = hOut.add_(1.0);
    out = out.add_(1.0);

    EXPECT_EQ(allclose(hOut.to(torch::kCPU), out, 0.001, 0.001), true);
  }

  // Restore hybrid sif flag
  SET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE, enable_fast_sif, 1);
  SET_ENV_FLAG_NEW(PT_HPU_RUN_HYBRID_SIF, run_hybrid_sif, 1);
}

TEST_F(LazyDynamicShapesTest, MatMulOutTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED)) {
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED, true, 1);
  }
  torch::Tensor B = torch::randn({2, 3, 2, 3, 7}, torch::dtype(torch::kFloat));
  torch::Tensor C = torch::randn({2, 3, 2, 7, 1}, torch::dtype(torch::kFloat));
  torch::Tensor out_cpu = torch::randn({0}, torch::dtype(torch::kFloat));
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor hC = C.to(torch::kHPU);
  torch::Tensor hOut = out_cpu.to(torch::kHPU);
  torch::matmul_out(hOut, hB, hC);
  torch::matmul_out(out_cpu, B, C);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), out_cpu, 0.001, 0.001), true);
  // Dyanamic shapes 1
  {
    torch::Tensor B =
        torch::randn({2, 12, 2, 3, 8}, torch::dtype(torch::kFloat));
    torch::Tensor C =
        torch::randn({2, 12, 2, 8, 1}, torch::dtype(torch::kFloat));
    torch::Tensor out_cpu = torch::randn({0}, torch::dtype(torch::kFloat));
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hC = C.to(torch::kHPU);
    torch::Tensor hOut = out_cpu.to(torch::kHPU);
    torch::matmul_out(hOut, hB, hC);
    torch::matmul_out(out_cpu, B, C);
    EXPECT_EQ(allclose(hOut.to(torch::kCPU), out_cpu, 0.001, 0.001), true);
  }
  // Dyanamic shapes 2
  {
    torch::Tensor B =
        torch::randn({2, 12, 2, 3, 7}, torch::dtype(torch::kFloat));
    torch::Tensor C =
        torch::randn({2, 12, 2, 7, 1}, torch::dtype(torch::kFloat));
    torch::Tensor out_cpu = torch::randn({0}, torch::dtype(torch::kFloat));
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hC = C.to(torch::kHPU);
    torch::Tensor hOut = out_cpu.to(torch::kHPU);
    torch::matmul_out(hOut, hB, hC);
    torch::matmul_out(out_cpu, B, C);
    EXPECT_EQ(allclose(hOut.to(torch::kCPU), out_cpu, 0.001, 0.001), true);
  }
  SET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED, false, 1);
}

/*
cat_out Op vaiant has issue.
It will be enable after fix of jira: SW-120927
*/
TEST_F(LazyDynamicShapesTest, CatOutTest) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  torch::Tensor A = torch::randn({2, 8}, torch::dtype(torch::kFloat));
  torch::Tensor B = torch::randn({4, 8}, torch::dtype(torch::kFloat));
  torch::Tensor out1 = torch::empty(0, at::kFloat);

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor h_out1 = out1.to(torch::kHPU);

  torch::cat_outf({A, B}, 0, out1);
  torch::cat_outf({hA, hB}, 0, h_out1);
  EXPECT_EQ(allclose(h_out1.to(torch::kCPU), out1, 0.001, 0.001), true);

  torch::Tensor X = torch::randn({4, 64566}, torch::dtype(torch::kFloat));
  torch::Tensor Y = torch::randn({8, 64566}, torch::dtype(torch::kFloat));
  torch::Tensor out2 = torch::empty(0, at::kFloat);

  torch::Tensor hX = X.to(torch::kHPU);
  torch::Tensor hY = Y.to(torch::kHPU);
  torch::Tensor h_out2 = out2.to(torch::kHPU);

  torch::cat_outf({X, Y}, 0, out2);
  torch::cat_outf({hX, hY}, 0, h_out2);
  EXPECT_EQ(allclose(h_out2.to(torch::kCPU), out2, 0.001, 0.001), true);
}

TEST_F(LazyDynamicShapesTest, H2D_API_test) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  if (false == GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED)) {
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED, true, 1);
  }

  std::vector<int64_t> in_sizes{6144, 24576, 98304};
  std::vector<std::vector<int64_t>> out_sizes{
      {2, 2, 32, 32}, {2, 2, 64, 64}, {2, 2, 128, 128}};
  std::vector<std::vector<int64_t>> strides{
      {3072, 1024, 32, 1}, {12288, 4096, 64, 1}, {49152, 16384, 128, 1}};
  std::vector<int64_t> offsets{1024, 4096, 16384};

  for (int i = 0; i < in_sizes.size(); i++) {
    auto in_s = in_sizes[i];
    c10::IntArrayRef out_s(out_sizes[i].data(), out_sizes[i].size());
    auto stride = strides[i];
    auto offset = offsets[i];

    PT_TEST_DEBUG("\n PTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({in_s});
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hOut = torch::as_strided(hA, out_s, stride, offset);
    torch::Tensor out = torch::as_strided(A, out_s, stride, offset);
    hOut = hOut.add_(1.0);
    out = out.add_(1.0);

    EXPECT_EQ(allclose(hOut.to(torch::kCPU), out, 0.001, 0.001), true);
  }

  SET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED, false, 1);
}

TEST_F(LazyDynamicShapesTest, gather_dynamic_test) {
  // static case
  torch::Tensor inp1 = torch::randn({2, 10, 53});
  auto inp2 = torch::randint(0, 4, {2, 10, 20}, torch::kLong);
  auto hinp1 = inp1.to(torch::kHPU);
  auto hinp2 = inp2.to(torch::kHPU);
  auto cpu = torch::gather(inp1, 2, inp2, 0);
  auto hpu = torch::gather(hinp1, 2, hinp2, 0);
  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(cpu, hpu.cpu()), true);
  // dynamic cache Miss
  {
    torch::Tensor inp1 = torch::randn({2, 100, 53});
    auto inp2 = torch::randint(0, 4, {2, 100, 5}, torch::kLong);
    auto hinp1 = inp1.to(torch::kHPU);
    auto hinp2 = inp2.to(torch::kHPU);
    auto cpu = torch::gather(inp1, 2, inp2, 0);
    auto hpu = torch::gather(hinp1, 2, hinp2, 0);
    HbLazyTensor::StepMarker({});
    EXPECT_EQ(allclose(cpu, hpu.cpu()), true);
  }
}

TEST_F(LazyDynamicShapesTest, MemCpy0D1D) {
  torch::Tensor A;
  for (auto i = 5; i < 10; i++) {
    A = (i < 8) ? torch::rand({i}) : torch::rand({});
    auto ones = torch::ones_like(A);

    auto hA = A.to(torch::kHPU);
    auto hOnes = ones.to(torch::kHPU);

    auto res = torch::add(A, ones);
    auto hRes = torch::add(hA, hOnes);

    auto out = torch::zeros_like(A);
    auto hOut = out.to(torch::kHPU);

    out.copy_(res);
    hOut.copy_(hRes);

    EXPECT_EQ(allclose(hOut.to(torch::kCPU), out, 0.001, 0.001), true);
  }
}

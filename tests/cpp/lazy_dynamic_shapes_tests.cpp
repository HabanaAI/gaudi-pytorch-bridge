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

class LazyDynamicShapesTest : public habana_lazy_test::LazyTest {};

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
  int kH = 3;
  int kW = 3;
  const int C = 16;
  const int N = 16;
  int H = 16;

  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

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
    torch::Tensor h_weight_tensor_hwck =
        h_weight_tensor.permute({2, 3, 1, 0}).contiguous();
    torch::Tensor h_out_conv =
        torch::conv2d(h_in_tensor, h_weight_tensor_hwck, {}, {1}, {0}, {1}, 1);
    torch::Tensor out_conv =
        torch::conv2d(in_tensor, weight_tensor, {}, {1}, {0}, {1}, 1);
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
  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
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

TEST_F(LazyDynamicShapesTest, DynamicShapeTest2) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  const int N = 16;
  int H = 16;

  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

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
    torch::Tensor h_weight_tensor_hwck =
        h_weight_tensor.permute({2, 3, 1, 0}).contiguous();
    torch::Tensor h_out_conv =
        torch::conv2d(h_in_tensor, h_weight_tensor_hwck, {}, {1}, {0}, {1}, 1);
    torch::Tensor out_conv =
        torch::conv2d(in_tensor, weight_tensor, {}, {1}, {0}, {1}, {1});
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
  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
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
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

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
        torch::randn(N * C * H * W, torch::requires_grad(false));
    torch::Tensor h_in_tensor = in_tensor.to(torch::kHPU);
    torch::Tensor h_weight_tensor_hwck =
        h_weight_tensor.permute({2, 3, 1, 0}).contiguous();
    torch::Tensor h_out_conv = torch::conv2d(
        h_in_tensor.reshape({N, C, H, W}),
        h_weight_tensor_hwck,
        {},
        {1},
        {0},
        {1},
        1);
    torch::Tensor out_conv = torch::conv2d(
        in_tensor.reshape({N, C, H, W}), weight_tensor, {}, {1}, {0}, {1}, 1);
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
  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

// Graph :
//
//     Bias1  Bias2           Data
//       \    /               |
//        \  /                |
//         Add                |
//          |-(weights)->  Convolution 3x3
//                            |
//                        Batch Norm
//                        Max Pool 2D           Bias3
//                           Relu             Broadcast
//                            |                  |
//                           Add <----------------
//                            |
//                         view/Reshape
//                            |
//                       UpSampleNearest2d
//                            |
//                           Add.Scalar(Constant)
//                            |
//                           out

TEST_F(LazyDynamicShapesTest, DISABLED_DynamicShapeTest4) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  const int N = 16;
  int H = 16;
  at::Scalar inScalar = 2.0;
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

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
    torch::Tensor h_weight_tensor_hwck =
        h_weight_tensor.permute({2, 3, 1, 0}).contiguous();
    torch::Tensor h_out_conv =
        torch::conv2d(h_in_tensor, h_weight_tensor_hwck, {}, {1}, {0}, {1}, 1);
    torch::Tensor out_conv =
        torch::conv2d(in_tensor, weight_tensor, {}, {1}, {0}, {1}, 1);
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
    auto h_out_upsample =
        torch::upsample_nearest2d(h_out_add, {}, scale_factors);
    auto out_upsample = torch::upsample_nearest2d(out_add, {}, scale_factors);
    // out = view(out_upsample)
    auto h_out_view = h_out_upsample.view({-1});
    auto out_view = out_upsample.view({-1});
    // out = Add(out_view,2)
    auto h_out = torch::add(h_out_view, inScalar);
    auto out = torch::add(out_view, inScalar);

    torch::Tensor out_hpu = h_out.to(torch::kCPU);
    EXPECT_EQ(allclose(out_hpu, out, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, DynamicShapeDebugSimple) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
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
  }
}

TEST_F(LazyDynamicShapesTest, SingleOpRelu) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

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

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, SetDynamicModeTest1) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  // Check org state of env flag
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  // unset the env variable if set for this case
  bool org_state = refine_enabled;
  if (refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
  refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  ASSERT_EQ(refine_enabled, false);
  HbLazyTensor::SetDynamicMode();
  // Check if env flag set for op Accumulation/execution
  refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
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
  // Check if env flag restored after execution
  refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  ASSERT_EQ(refine_enabled, false);
  torch::Tensor sum_hpu = sum_tensor.to(torch::kCPU);
  EXPECT_EQ(allclose(sum_cpu, sum_hpu, 0.01, 0.01), true);
  if (org_state) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
}

TEST_F(LazyDynamicShapesTest, SetDynamicModeTest2) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  // Set the env flag if not set
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  bool org_state = refine_enabled;
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
  // Check org state of env flag
  refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  ASSERT_EQ(refine_enabled, true);
  HbLazyTensor::SetDynamicMode();
  // Check if env flag set for op Accumulation/execution
  refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
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
  refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  ASSERT_EQ(refine_enabled, true);
  torch::Tensor sum_hpu = sum_tensor.to(torch::kCPU);
  EXPECT_EQ(allclose(sum_cpu, sum_hpu, 0.01, 0.01), true);
  // restore the original env variable
  if (!org_state) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, SetDynamicModeTest3) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  // Check org state of env flag
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  // unset the env variable if set for this case
  bool org_state = refine_enabled;
  if (refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
  // Check if dynamic mode is unset
  refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
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
  refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  ASSERT_EQ(refine_enabled, false);
  torch::Tensor sum_hpu = sum_tensor.to(torch::kCPU);
  EXPECT_EQ(allclose(sum_cpu, sum_hpu, 0.01, 0.01), true);
  // restore the original state
  if (org_state) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
}

TEST_F(LazyDynamicShapesTest, DynamicAvgPoolBkwdTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
  int N = 1;
  const int C = 16;
  int H = 16;
  std::vector<int> in_sizes{16, 32, 64};

  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    auto input_tensor = torch::randn({N, C, H, W}, torch::requires_grad(true));
    auto cpu_pool = torch::avg_pool2d(input_tensor, 3, 1);
    auto cpu_out = torch::relu(cpu_pool);

    // fwd propagation
    torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
    auto outHabana1 =
        torch::avg_pool2d(tHabanaX, {3, 3}, {1, 1}, {0, 0}, false, true);
    torch::Tensor outHabana = torch::relu(outHabana1);

    // bwd propagation with dummy grad tensor
    auto grad_tensor =
        torch::randn({N, C, H - 2, W - 2}, torch::requires_grad(true));
    torch::Tensor tHabanaG = grad_tensor.to(torch::kHPU);
    outHabana.backward({tHabanaG}, false, true);

    auto out_cpu_lazy = outHabana.to(torch::kCPU);
    ASSERT_TRUE(torch::allclose(out_cpu_lazy, cpu_out));
  }
  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, DynamicMaxPoolBkwdTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
  int N = 1;
  const int C = 16;
  int H = 16;
  std::vector<int> in_sizes{16, 32, 64};

  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    auto input_tensor = torch::randn({N, C, H, W}, torch::requires_grad(true));
    auto cpu_pool = torch::max_pool2d(input_tensor, 3, 1);
    auto cpu_out = torch::relu(cpu_pool);

    // fwd propgation
    torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
    auto outHabana1 = torch::max_pool2d_with_indices(
        tHabanaX, {3, 3}, {1, 1}, {0, 0}, {1, 1}, true);
    torch::Tensor outHabana = torch::relu(std::get<0>(outHabana1));

    // bwd propgation with dummy grad tensor
    auto grad_tensor =
        torch::randn({N, C, H - 2, W - 2}, torch::requires_grad(true));
    torch::Tensor tHabanaG = grad_tensor.to(torch::kHPU);
    outHabana.backward({tHabanaG}, false, true);

    auto out_cpu_lazy = outHabana.to(torch::kCPU);
    ASSERT_TRUE(torch::allclose(out_cpu_lazy, cpu_out));
  }
  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, DynamicConvBkwdTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
  int kH = 3;
  int kW = 3;
  int N = 1;
  const int C = 3;
  int H = 6;
  std::vector<int> in_sizes{3, 6, 9};

  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor weight_tensor =
        torch::randn({C, C, kW, kH}, torch::requires_grad(true));
    auto in_tensor = torch::randn({N, C, H, W}, torch::requires_grad(true));
    // cpu
    torch::Tensor out_conv =
        torch::conv2d(in_tensor, weight_tensor, {}, {1}, {0}, {1}, 1);
    auto cpu_out = torch::relu(out_conv);

    // fwd propgation
    torch::Tensor h_weight_tensor = weight_tensor.to(torch::kHPU);
    torch::Tensor h_weight_tensor_hwck =
        h_weight_tensor.permute({2, 3, 1, 0}).contiguous();
    torch::Tensor h_in_tensor = in_tensor.to(torch::kHPU);
    torch::Tensor h_out_conv =
        torch::conv2d(h_in_tensor, h_weight_tensor_hwck, {}, {1}, {0}, {1}, 1);
    torch::Tensor hpu_out = torch::relu(h_out_conv);

    // bwd propgation with dummy grad tensor
    auto grad_tensor =
        torch::randn({N, C, H - 2, W - 2}, torch::requires_grad(true));
    torch::Tensor tHabanaG = grad_tensor.to(torch::kHPU);
    hpu_out.backward({tHabanaG}, false, true);

    auto out_cpu_lazy = hpu_out.to(torch::kCPU);
    ASSERT_TRUE(torch::allclose(out_cpu_lazy, cpu_out));
  }
  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, ProdTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

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
  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, SliceTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
  int N = 1;
  int C = 4;
  int H = 4;
  std::vector<int> in_sizes{16, 18, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    int64_t dim = 2;
    int64_t start_index = 0;
    int64_t end = 3;
    int64_t step = 1;

    torch::Tensor h_out = torch::slice(hA, dim, start_index, end, step);

    auto h_cout = h_out.to(torch::kCPU);
    auto cout = torch::slice(A, dim, start_index, end, step);

    EXPECT_EQ(allclose(h_cout, cout), true);
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, SliceTest2) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
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
    int64_t step = 1;

    torch::Tensor h_out = torch::slice(hA, dim, start_index, end, step);

    auto h_cout = h_out.to(torch::kCPU);
    auto cout = torch::slice(A, dim, start_index, end, step);

    EXPECT_EQ(allclose(h_cout, cout), true);
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, ExpandTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

  std::vector<int> W_in_sizes{1, 482, 1, 482, 1, 482};
  std::vector<int> H_in_sizes{200, 1, 200, 1, 1, 200};
  for (int i = 0; i < W_in_sizes.size(); i++) {
    int W = W_in_sizes[i];
    int H = H_in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({W, H}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);

    torch::Tensor h_out = hA.expand({482, 200});

    auto h_cout = h_out.to(torch::kCPU);
    auto cout = A.expand({482, 200});

    EXPECT_EQ(allclose(h_cout, cout), true);
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, ExpandTest2) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
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

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, RepeatTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

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

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, RepeatTest2) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

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

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, DynamicShapeInplaceTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
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

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, ArangeTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

  // std::vector<int> start_sizes{1, 1, 1, 1};
  std::vector<int> start_sizes{0, 2, 3, 4};
  std::vector<int> end_sizes{5, 10, 15, 18};
  std::vector<int> step_sizes{1, 2, 3, 2};
  for (int i = 0; i < start_sizes.size(); i++) {
    torch::Scalar start = start_sizes[i];
    torch::Scalar end = end_sizes[i];
    torch::Scalar step = step_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");

    c10::optional<at::ScalarType> dtype = c10::ScalarType::Int;

    c10::optional<at::Device> hb_device = at::DeviceType::HPU;
    at::TensorOptions hb_options =
        at::TensorOptions().dtype(dtype).device(hb_device);
    c10::optional<at::Device> cpu_device = at::DeviceType::CPU;
    at::TensorOptions cpu_options =
        at::TensorOptions().dtype(dtype).device(cpu_device);

    auto h_a = torch::arange(start, end, step, hb_options);
    auto h_cout = h_a.to(torch::kCPU);
    auto a = torch::arange(start, end, step, cpu_options);
    EXPECT_EQ(allclose(h_cout, a), true);
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, DynamicShapeInplaceTest2) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

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

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, DynamicShapeInplaceReluTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

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

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, AddConstantTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
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

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, AddViewTest) {
  // test case for result = add(tensor, scalar, alpha)
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
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

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, CastTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
  PT_TEST_DEBUG("\nPTI_DBG :: TEST ", 0, "  --------\n");
  torch::Tensor A = torch::randn({1}, torch::dtype(torch::kBFloat16));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = hA.to(torch::kFloat);
  torch::Tensor Out = A.to(torch::kFloat);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.001, 0.001), true);
  int H = 1024;
  std::vector<int> in_sizes{32768, 65536};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i + 1, "  --------\n");
    torch::Tensor A = torch::randn({W, H}, torch::dtype(torch::kBFloat16));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hOut = hA.to(torch::kFloat);
    torch::Tensor Out = A.to(torch::kFloat);
    EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.001, 0.001), true);
  }
  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, UniqueOp) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

  c10::ScalarType dtype{torch::kInt32};

  std::vector<int> in_sizes{4, 6, 8};
  for (int i = 0; i < in_sizes.size(); i++) {
    int H = 4;
    int W = in_sizes[i];
    torch::Tensor input_cpu = torch::randint(1, 9, {H, W}, dtype);
    torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
    auto out_hpu = std::get<0>(torch::_unique2(input_hpu, false, false, false));
    auto out_cpu = std::get<0>(torch::_unique2(input_cpu, false, false, false));
    PRINT_TENSOR_DETAILS(out_cpu);
    auto h_cout = out_hpu.to(torch::kCPU);
    EXPECT_EQ(
        allclose(
            std::get<0>(h_cout.view(-1).sort()),
            std::get<0>(out_cpu.view(-1).sort())),
        true);

    auto out_cpuv = std::get<0>(out_cpu.view(-1).sort());
    auto out_hpuv = std::get<0>(h_cout.view(-1).sort());
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, SingleOpNonzero) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

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

    torch::Tensor out_cpu = torch::nonzero(c0).to(torch::kInt32);

    PRINT_TENSOR_DETAILS(c0);
    PRINT_TENSOR_DETAILS(out_cpu);

    torch::Tensor h0 = c0.to(torch::kHPU);
    torch::Tensor out_hpu = torch::nonzero(h0);
    torch::Tensor out_hpu_c = out_hpu.to(torch::kCPU);

    PRINT_TENSOR_DETAILS(h0);
    PRINT_TENSOR_DETAILS(out_hpu_c);

    EXPECT_EQ(allclose(out_cpu, out_hpu_c, 0.01, 0.01), true);
    PT_TEST_DEBUG("TEST ", i, "  ========");
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
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
  float score_th = 0.1;
  float score_inc = 0.2;

  auto num_boxes = 10;
  auto num_boxes_fixed = 8;
  auto num_boxes_variable = num_boxes - num_boxes_fixed;
  torch::Tensor scores = torch::rand({num_boxes});
  torch::Tensor boxes_fixed = torch::rand({num_boxes_fixed, 4}) * 256;

  while (score_th < 1.0) {
    auto num_expected_boxes{0};
    for (size_t i = 0; i < num_boxes; i++) {
      float score = scores[i].item<float>();
      if (score > score_th) {
        num_expected_boxes++;
      }
    }
    if (num_expected_boxes) {
      torch::Tensor hscores = scores.to(torch::kHPU);

      // Generate boxes of random sizes
      torch::Tensor boxes_variable = torch::rand({num_boxes_variable, 4}) * 256;
      torch::Tensor boxes = torch::cat({boxes_fixed, boxes_variable}, 0);

      // Ensure x1 > x0 and y1 > y0
      auto tlist = boxes.split(2, 1);
      tlist[1] = tlist[1] + tlist[0];
      auto valid_boxes = torch::cat({tlist[0], tlist[1]}, 1);
      // PRINT_TENSOR_DETAILS(valid_boxes);

      // Compute the iou scores
      // std::vector<std::vector<float>> iou_vec_2d;
      // iou_vec_2d.reserve(num_boxes-1);
      // compute_iou(valid_boxes, iou_vec_2d);

      torch::Tensor hboxes = valid_boxes.to(torch::kHPU);

      auto nms_boxid = habana_nms_hpu_lazy(hboxes, hscores, 1.0, score_th);
      auto nms_boxid_c = nms_boxid.to(torch::kCPU);
      TORCH_CHECK(
          nms_boxid_c.dim() == 1,
          "Expecting a 1D tensor, got ",
          boxes.dim(),
          "D tensor");
      PT_TEST_DEBUG(
          "With score threshold=",
          score_th,
          ", num_expected_boxes=",
          num_expected_boxes,
          ", got ",
          nms_boxid_c.sizes()[0]);
      auto equal = (nms_boxid_c.sizes()[0] == num_expected_boxes);
      EXPECT_EQ(equal, true);
    }
    score_th += score_inc;
  }
}

TEST_F(LazyDynamicShapesTest, NmsSmall) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

  torch::manual_seed(0);
  float score_th = 0.5;
  float score_inc = 0.2;

  auto num_boxes_cur = 8;
  auto num_boxes_var = 2;
  torch::Tensor scores_cur = torch::rand({num_boxes_cur});
  torch::Tensor boxes_cur = torch::rand({num_boxes_cur, 4}) * 256;

  while (num_boxes_cur < 13) {
    PRINT_TENSOR_DETAILS(boxes_cur);
    PRINT_TENSOR_DETAILS(scores_cur);

    auto num_expected_boxes{0};
    for (size_t i = 0; i < num_boxes_cur; i++) {
      float score = scores_cur[i].item<float>();
      if (score > score_th) {
        num_expected_boxes++;
      }
    }

    if (num_expected_boxes) {
      torch::Tensor hscores = scores_cur.to(torch::kHPU);

      // Ensure x1 > x0 and y1 > y0
      auto tlist = boxes_cur.split(2, 1);
      tlist[1] = tlist[1] + tlist[0];
      auto valid_boxes = torch::cat({tlist[0], tlist[1]}, 1);
      // PRINT_TENSOR_DETAILS(valid_boxes);

      // Compute the iou scores
      // std::vector<std::vector<float>> iou_vec_2d;
      // iou_vec_2d.reserve(num_boxes_cur-1);
      // compute_iou(valid_boxes, iou_vec_2d);

      torch::Tensor hboxes = valid_boxes.to(torch::kHPU);

      auto nms_boxid = habana_nms_hpu_lazy(hboxes, hscores, 1.0, score_th);
      auto nms_boxid_c = nms_boxid.to(torch::kCPU);
      TORCH_CHECK(
          nms_boxid_c.dim() == 1,
          "Expecting a 1D tensor, got ",
          boxes_cur.dim(),
          "D tensor");
      // PRINT_TENSOR_DETAILS(scores_cur);
      // PRINT_TENSOR_DETAILS(nms_boxid_c);
      PT_TEST_DEBUG(
          "With score threshold=",
          score_th,
          ", num_expected_boxes=",
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
  // while (score_th < 1.0) {
  // score_th += score_inc;
  //}

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, ArgmaxTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
  int N = 1;
  int C = 4;
  int H = 4;
  std::vector<int> in_sizes{6, 12, 20, 10};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({N, C, H, W}, torch::requires_grad(false));

    torch::Tensor hA = A.to(torch::kHPU);

    torch::Tensor out_hpu = torch::argmax(hA, 2);
    torch::Tensor out_cpu = torch::argmax(A, 2);
    auto out = out_hpu.to(torch::kCPU);
    EXPECT_TRUE(allclose(out, out_cpu.to(torch::kInt), 0.0001, 0.0001));
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, ViewTest) {
  // test case for result = add(tensor, scalar, alpha)
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }
  int N = 2;
  int C = 4;
  int H = 4;
  at::Scalar alpha = 1.0;
  at::Scalar Y = 2.0;
  std::vector<int> in_sizes{6, 8, 10};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    std::vector<int64_t> shape{N, C, H * W, 1};
    torch::Tensor C = A.reshape(c10::IntArrayRef(shape));
    torch::Tensor hC = hA.reshape(c10::IntArrayRef(shape));
    auto C_out = hC.to(torch::kCPU);
    EXPECT_EQ(allclose(C, C_out, 0.001, 0.001), true);
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, MaskRcnnGatherNdMxNetTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

  int64_t dim = 0;
  int H = 4;
  std::vector<int> in_sizes{8000, 9000, 10000};

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
    indices_cpu.push_back(
        c10::make_optional(torch::slice(index, 0, 0, 1000, 1)));
    indices_list.push_back(
        c10::make_optional(torch::slice(index.to(torch::kHPU), 0, 0, 1000, 1)));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hOut = torch::index(hA, indices_list);
    torch::Tensor out = torch::index(A, indices_cpu);
    torch::Tensor hOut1 = torch::index(hB, indices_list);
    torch::Tensor out1 = torch::index(B, indices_cpu);
    HbLazyTensor::StepMarker({});
    EXPECT_EQ(allclose(hOut.to(torch::kCPU), out, 0.001, 0.001), true);
  }
  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

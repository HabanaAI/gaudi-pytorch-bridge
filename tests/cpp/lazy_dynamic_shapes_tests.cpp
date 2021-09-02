/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include "habana_lazy_test_infra.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

using namespace habana_lazy;

class LazyDynamicShapesTest : public habana_lazy_test::LazyTest {};

// This test is testing dynamic shapes milestone 2 logic. Check out
// [SW-20642] in the jira to find out the full test
// layout. But simply its doing :
//
//     Bias1  Bias2           Data
//       \    /               |
//         Add                |
//          |-(weights)->  Convolution 3x3
//                            |
//                        Batch Norm
//                        Avg Pool 2D           Bias3
//                           Relu             Broadcast
//                            |                  |
//                           Add <----------------
//                            |
//                           out

const char cinTerminator = 'q';
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
    std::cout << "PTI_DBG: Iteration Start -- " << i << " ----\n";
    int W = in_sizes[i];
    // weight_tensor = bias1 + bias2
    torch::Tensor bias1 =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    torch::Tensor bias2 =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    torch::Tensor h_bias1 = bias1.to(torch::kHABANA);
    torch::Tensor h_bias2 = bias2.to(torch::kHABANA);
    torch::Tensor weight_tensor = torch::add(bias1, bias2);
    torch::Tensor h_weight_tensor = torch::add(h_bias1, h_bias2);
    // out_conv = Conv3x3(Data, weight)
    torch::Tensor in_tensor =
        torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor h_in_tensor = in_tensor.to(torch::kHABANA);
    torch::Tensor h_weight_tensor_hwck =
        h_weight_tensor.permute({2, 3, 1, 0}).contiguous();
    torch::Tensor h_out_conv =
        torch::conv2d(h_in_tensor, h_weight_tensor_hwck, {}, 1, 0, 1, 1);
    torch::Tensor out_conv =
        torch::conv2d(in_tensor, weight_tensor, {}, 1, 0, 1, 1);
    // bn_out = BatchNorm(out_conv)
    torch::Tensor gamma =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor beta =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor mean =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor var =
        torch::ones(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor h_gamma = gamma.to(torch::kHABANA);
    torch::Tensor h_beta = beta.to(torch::kHABANA);
    torch::Tensor h_mean = mean.to(torch::kHABANA);
    torch::Tensor h_var = var.to(torch::kHABANA);
    float mom = 0.1;
    float eps = 1e-5;
    auto h_bn_outs = torch::native_batch_norm(
        h_out_conv, h_gamma, h_beta, h_mean, h_var, false, mom, eps);
    auto bn_outs = torch::native_batch_norm(
        out_conv, gamma, beta, mean, var, false, mom, eps);
    auto h_bn_out = std::get<0>(h_bn_outs);
    auto bn_out = std::get<0>(bn_outs);
    // TODO: Enable this after adding shape Tensor to MaxPool Kernel
    // pool_out = MaxPool2D(bn_out)
    // auto h_pool_outs = torch::max_pool2d_with_indices(
    //     h_bn_out, {2, 2}, {2, 2}, {0, 0}, {1, 1}, true);
    // torch::Tensor h_pool_out = std::get<0>(h_pool_outs);
    // torch::Tensor pool_out = torch::max_pool2d(bn_out, 2, 2);
    // pool_out = avg_pool2d(bn_out)
    torch::Tensor pool_out = torch::avg_pool2d(bn_out, 3, 1);
    torch::Tensor h_pool_out =
        torch::avg_pool2d(h_bn_out, {3, 3}, {1, 1}, {0, 0}, false, true);
    // relu_out = relu(pool_out)
    torch::Tensor h_relu_out = torch::relu(h_pool_out);
    torch::Tensor relu_out = torch::relu(pool_out);
    // out = add(relu_out, x)
    torch::Tensor bias3 =
        torch::randn(1, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor h_bias3 = bias3.to(torch::kHABANA);
    auto h_out = torch::add(h_relu_out, h_bias3);
    auto out = torch::add(relu_out, bias3);

    torch::Tensor out_hpu = h_out.to(torch::kCPU);
    EXPECT_EQ(allclose(out_hpu, out, 0.01, 0.01), true);
    std::cout << "PTI_DBG: Iteration End -- " << i << " ----\n";
  }
  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

// This test is testing dynamic shapes milestone 2 logic. Check out
// [SW-20642] in the jira to find out the full test
// layout. But simply its doing :
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
    std::cout << "PTI_DBG: Iteration Start -- " << i << " ----\n";
    int W = in_sizes[i];
    // weight_tensor = bias1 + bias2
    torch::Tensor bias1 =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    torch::Tensor bias2 =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    torch::Tensor h_bias1 = bias1.to(torch::kHABANA);
    torch::Tensor h_bias2 = bias2.to(torch::kHABANA);
    torch::Tensor weight_tensor = torch::add(bias1, bias2);
    torch::Tensor h_weight_tensor = torch::add(h_bias1, h_bias2);
    // out_conv = Conv3x3(Data, weight)
    torch::Tensor in_tensor =
        torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor h_in_tensor = in_tensor.to(torch::kHABANA);
    torch::Tensor h_weight_tensor_hwck =
        h_weight_tensor.permute({2, 3, 1, 0}).contiguous();
    torch::Tensor h_out_conv =
        torch::conv2d(h_in_tensor, h_weight_tensor_hwck, {}, 1, 0, 1, 1);
    torch::Tensor out_conv =
        torch::conv2d(in_tensor, weight_tensor, {}, 1, 0, 1, 1);
    // bn_out = BatchNorm(out_conv)
    torch::Tensor gamma =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor beta =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor mean =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor var =
        torch::ones(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor h_gamma = gamma.to(torch::kHABANA);
    torch::Tensor h_beta = beta.to(torch::kHABANA);
    torch::Tensor h_mean = mean.to(torch::kHABANA);
    torch::Tensor h_var = var.to(torch::kHABANA);
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
    torch::Tensor h_bias3 = bias3.to(torch::kHABANA);
    auto h_out = torch::add(h_relu_out, h_bias3);
    auto out = torch::add(relu_out, bias3);

    torch::Tensor out_hpu = h_out.to(torch::kCPU);
    EXPECT_EQ(allclose(out_hpu, out, 0.01, 0.01), true);
    std::cout << "PTI_DBG: Iteration End -- " << i << " ----\n";
  }
  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

/*
 * TEST HAS BEEN DISABLED UNTILL WE HAVE SUPPORT FOR
 * SHAPE INFERENCE FROM GC IS ENABLED
 */
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
    std::cout << '\n';
    std::cout << "PTI_DBG :: TEST " << i << "  --------" << '\n';
    torch::Tensor c0 = torch::randn({C, B, A}, torch::requires_grad(false));
    torch::Tensor c1 = torch::randn({C, B, A}, torch::requires_grad(false));

    torch::Tensor c4 = torch::add(c0, c1);
    torch::Tensor c5 = torch::mul(c0, c1);
    torch::Tensor c6 = torch::mul(c4, c5);
    torch::Tensor c7 = torch::relu(c6);

    std::cout << "PTI_DBG ::"
              << " c0.shape : " << c0.sizes()
              << " c0.strides : " << c0.strides() << '\n';
    std::cout << "PTI_DBG ::"
              << " c1.shape : " << c1.sizes()
              << " c1.strides : " << c1.strides() << '\n';

    // std::cout << "PTI_DBG ::" << " c4.shape : " << c4.sizes() << " c4.strides
    // : " << c4.strides() << '\n'; std::cout << "PTI_DBG ::" << " c5.shape : "
    // << c5.sizes() << " c5.strides : " << c5.strides() << '\n'; std::cout <<
    // "PTI_DBG ::" << " c6.shape : " << c6.sizes() << " c6.strides : " <<
    // c6.strides() << '\n';
    //
    std::cout << "PTI_DBG ::"
              << " c7.shape : " << c7.sizes()
              << " c7.strides : " << c7.strides() << '\n';

    torch::Tensor h0 = c0.to(torch::kHABANA);
    torch::Tensor h1 = c1.to(torch::kHABANA);
    torch::Tensor h4 = torch::add(h0, h1);
    torch::Tensor h5 = torch::mul(h0, h1);
    torch::Tensor h6 = torch::mul(h4, h5);
    torch::Tensor h7 = torch::relu(h6);
    torch::Tensor h7_c = h7.to(torch::kCPU);

    std::cout << "PTI_DBG ::"
              << " h0.shape : " << h0.sizes()
              << " h0.strides : " << h0.strides() << '\n';
    std::cout << "PTI_DBG ::"
              << " h1.shape : " << h1.sizes()
              << " h1.strides : " << h1.strides() << '\n';

    std::cout << "PTI_DBG ::"
              << " h7.shape : " << h7.sizes()
              << " h7.strides : " << h7.strides() << '\n';

    EXPECT_EQ(allclose(c7, h7_c, 0.01, 0.01), true);
    std::cout << "PTI_DBG :: TEST " << i << "  ========" << '\n';
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, DynamicShapeDebugSimple2) {
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
    std::cout << '\n';
    std::cout << "PTI_DBG :: TEST " << i << "  --------" << '\n';
    torch::Tensor c0 = torch::randn({C, B, A}, torch::requires_grad(false));

    torch::Tensor c4 = torch::relu(c0);

    std::cout << "PTI_DBG ::"
              << " c0.shape : " << c0.sizes()
              << " c0.strides : " << c0.strides() << '\n';

    std::cout << "PTI_DBG ::"
              << " c4.shape : " << c4.sizes()
              << " c4.strides : " << c4.strides() << '\n';

    torch::Tensor h0 = c0.to(torch::kHABANA);
    torch::Tensor h4 = torch::relu(h0);
    torch::Tensor h4_c = h4.to(torch::kCPU);

    std::cout << "PTI_DBG ::"
              << " h0.shape : " << h0.sizes()
              << " h0.strides : " << h0.strides() << '\n';

    std::cout << "PTI_DBG ::"
              << " h4.shape : " << h4.sizes()
              << " h4.strides : " << h4.strides() << '\n';

    EXPECT_EQ(allclose(c4, h4_c, 0.01, 0.01), true);
    std::cout << "PTI_DBG :: TEST " << i << "  ========" << '\n';
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
  torch::Tensor h_num1 = num1.to(torch::kHABANA);
  torch::Tensor h_num2 = num2.to(torch::kHABANA);
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
  torch::Tensor h_num1 = num1.to(torch::kHABANA);
  torch::Tensor h_num2 = num2.to(torch::kHABANA);
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
  torch::Tensor h_num1 = num1.to(torch::kHABANA);
  torch::Tensor h_num2 = num2.to(torch::kHABANA);
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
    std::cout << '\n';
    std::cout << "PTI_DBG :: TEST " << i << "  --------" << '\n';
    auto input_tensor = torch::randn({N, C, H, W}, torch::requires_grad(true));
    auto cpu_pool = torch::avg_pool2d(input_tensor, 3, 1);
    auto cpu_out = torch::relu(cpu_pool);

    // fwd propagation
    torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
    auto outHabana1 =
        torch::avg_pool2d(tHabanaX, {3, 3}, {1, 1}, {0, 0}, false, true);
    torch::Tensor outHabana = torch::relu(outHabana1);

    // bwd propagation with dummy grad tensor
    auto grad_tensor =
        torch::randn({N, C, H - 2, W - 2}, torch::requires_grad(true));
    torch::Tensor tHabanaG = grad_tensor.to(torch::kHABANA);
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
    std::cout << '\n';
    std::cout << "PTI_DBG :: TEST " << i << "  --------" << '\n';
    auto input_tensor = torch::randn({N, C, H, W}, torch::requires_grad(true));
    auto cpu_pool = torch::max_pool2d(input_tensor, 3, 1);
    auto cpu_out = torch::relu(cpu_pool);

    // fwd propgation
    torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
    auto outHabana1 = torch::max_pool2d_with_indices(
        tHabanaX, {3, 3}, {1, 1}, {0, 0}, {1, 1}, true);
    torch::Tensor outHabana = torch::relu(std::get<0>(outHabana1));

    // bwd propgation with dummy grad tensor
    auto grad_tensor =
        torch::randn({N, C, H - 2, W - 2}, torch::requires_grad(true));
    torch::Tensor tHabanaG = grad_tensor.to(torch::kHABANA);
    outHabana.backward({tHabanaG}, false, true);

    auto out_cpu_lazy = outHabana.to(torch::kCPU);
    ASSERT_TRUE(torch::allclose(out_cpu_lazy, cpu_out));
  }
  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

TEST_F(LazyDynamicShapesTest, DISABLED_ProdTest) {
  bool refine_enabled = GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    setenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "1", 1);
  }

  int H = 4;
  std::vector<int> in_sizes{6, 8, 10};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    std::cout << '\n';
    std::cout << "PTI_DBG :: TEST " << i << "  --------" << '\n';
    torch::Tensor A = torch::randn({H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHABANA);
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
    std::cout << '\n';
    std::cout << "PTI_DBG :: TEST " << i << "  --------" << '\n';
    torch::Tensor A = torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHABANA);
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
    std::cout << '\n';
    std::cout << "PTI_DBG :: TEST " << i << "  --------" << '\n';
    torch::Tensor c0 = torch::randn({A, B});
    torch::Tensor c1 = torch::randn({A, B});
    torch::Tensor c2 = torch::randn({A, B});

    torch::Tensor h0 = c0.to(torch::kHABANA);
    torch::Tensor h1 = c1.to(torch::kHABANA);
    torch::Tensor h2 = c2.to(torch::kHABANA);

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
    std::cout << '\n';
    std::cout << "PTI_DBG :: TEST " << i << "  --------" << '\n';
    torch::Tensor c0 = torch::randn({A, B});
    torch::Tensor c1 = torch::randn({A, B});
    torch::Tensor c2 = torch::randn({A, B});

    torch::Tensor h0 = c0.to(torch::kHABANA);
    torch::Tensor h1 = c1.to(torch::kHABANA);
    torch::Tensor h2 = c2.to(torch::kHABANA);

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
    std::cout << '\n';
    std::cout << "PTI_DBG :: TEST " << i << "  --------" << '\n';
    torch::Tensor c0 = torch::randn({C, B, A}, torch::requires_grad(false));

    c0 = torch::relu_(c0);

    torch::Tensor h0 = c0.to(torch::kHABANA);

    h0 = torch::relu_(h0);
    torch::Tensor h0_c = h0.to(torch::kCPU);

    EXPECT_EQ(allclose(c0, h0_c, 0.01, 0.01), true);
  }

  if (!refine_enabled) {
    unsetenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES");
  }
}

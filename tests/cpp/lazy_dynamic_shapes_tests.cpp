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
//                        Max Pool 2D           Bias3
//                           Relu             Broadcast
//                            |                  |
//                           Add <----------------
//                            |
//                           out

const char cinTerminator = 'q';
TEST_F(LazyDynamicShapesTest, DISABLED_DynamicShapeTest) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  const int BATCH = 16;
  std::vector<int> in_sizes;
  int num;
  std::cout << "Enter Input sizes(W=H), q to terminate and execute"
            << std::endl;
  while ((std::cin >> num) && num != cinTerminator) {
    in_sizes.push_back(num);
    std::cout << "Enter Input sizes(W=H), q to terminate and execute"
              << std::endl;
  }

  for (int i = 0; i < in_sizes.size(); i++) {
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
    torch::Tensor in_tensor = torch::randn(
        {BATCH, C, in_sizes[i], in_sizes[i]}, torch::requires_grad(false));
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
    torch::Tensor h_bias3 = bias3.to(torch::kHABANA);
    auto h_out = torch::add(h_relu_out, h_bias3);
    auto out = torch::add(relu_out, bias3);

    torch::Tensor out_hpu = h_out.to(torch::kCPU);
    EXPECT_EQ(allclose(out_hpu, out, 0.01, 0.01), true);
  }
}

TEST_F(LazyDynamicShapesTest, DISABLED_DynamicShape3DTensorBasicGraphTest) {
  std::vector<int> dims;

  int num_runs;
  std::cout << "Enter number of runs" << std::endl;
  std::cin >> num_runs;

  for (int i = 0; i < num_runs; ++i) {
    // Get the 3d tensor shapes
    int dim_shape;
    dims.clear();
    std::cout << "Enter dim1\n";
    std::cin >> dim_shape;
    dims.push_back(dim_shape);
    std::cout << "Enter dim2\n";
    std::cin >> dim_shape;
    dims.push_back(dim_shape);
    std::cout << "Enter dim3\n";
    std::cin >> dim_shape;
    dims.push_back(dim_shape);

    torch::Tensor in1 =
        torch::randn({dims[0], dims[1], dims[2]}, torch::requires_grad(false));
    torch::Tensor in2 =
        torch::randn({dims[0], dims[1], dims[2]}, torch::requires_grad(false));
    torch::Tensor h_in1 = in1.to(torch::kHABANA);
    torch::Tensor h_in2 = in2.to(torch::kHABANA);

    torch::Tensor add_out = torch::add(in1, in2);
    torch::Tensor h_add_out = torch::add(h_in1, h_in2);
    torch::Tensor relu_out = torch::relu(add_out);
    torch::Tensor h_relu_out = torch::relu(h_add_out);
    torch::Tensor mul_out = torch::mul(relu_out, in2);
    torch::Tensor h_mul_out = torch::mul(h_relu_out, h_in2);
    auto out = torch::abs(mul_out);
    auto h_out = torch::abs(h_mul_out);

    torch::Tensor out_hpu = h_out.to(torch::kCPU);
    EXPECT_EQ(allclose(out_hpu, out, 0.01, 0.01), true);
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

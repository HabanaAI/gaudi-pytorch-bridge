#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

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
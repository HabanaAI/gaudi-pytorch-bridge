/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_lazy_test_infra.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include "habana_kernels/lazy_kernels_declarations.h"

#include "pytorch_helpers/habana_helpers/dynamic_bucket_info.h"
#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/habana_helpers/tensor_utils.h"

#include "pytorch_helpers/synapse_helpers/env_flags.h"

using namespace habana_lazy;

// In this class both the pass fallback and compilation fallback are disabled
class LazyDynamicComputeOutputShapesTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    SetLazyMode();

    SetSeed();

    DisableCpuFallback();

    SetDynamicMode();

    DisableDynamicPassFallback();

    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();

    habana::RecipeCacheLRU::get_cache().clear();
  }

  void TearDown() override {
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();

    UnsetDynamicMode();

    RestoreDynamicPassFallback();

    RestoreMode();
  }
};

// Test Add Add Div Sub
TEST_F(LazyDynamicComputeOutputShapesTest, AddAddDivSub) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  const int H = 8;
  const int C = 4;
  const int N = 2;

  std::vector<int> in_sizes{8, 16, 32};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    int W = in_sizes[i];
    const std::vector<int64_t> dimentions{N, C, H, W};
    torch::Tensor A = torch::randn(dimentions);
    torch::Tensor B = torch::randn(dimentions);
    torch::Tensor C = torch::randn(dimentions);
    torch::Tensor D = torch::randn(dimentions);
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hC = C.to(torch::kHPU);
    torch::Tensor hD = D.to(torch::kHPU);
    torch::Tensor add_out1 = torch::add(hA, hB, 2.3);
    torch::Tensor add_out2 = torch::add(hC, add_out1, 3.4);
    torch::Tensor div_out3 = torch::div(add_out2, 6);
    torch::Tensor out = torch::sub(div_out3, hD);

    torch::Tensor add_out1_cpu = torch::add(A, B, 2.3);
    torch::Tensor add_out2_cpu = torch::add(C, add_out1_cpu, 3.4);
    torch::Tensor div_out3_cpu = torch::div(add_out2_cpu, 6);
    torch::Tensor out_cpu = torch::sub(div_out3_cpu, D);

    EXPECT_EQ(allclose(out.to(torch::kCPU), out_cpu, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Graph :
//
//     Bias1  Bias2           Data
//       \    /               |
//         Add                |
//          |-(weights)->  Convolution 3x3
//                            |
//                        Batch Norm
//                        Max Pool 2D           Bias3
//                            |              Broadcast
//                            |                  |
//                           Add <----------------
//                            |
//                           Out
TEST_F(LazyDynamicComputeOutputShapesTest, AddConv2DBNMaxPoolTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
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
    if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) &&
        !habana_lazy::exec::OptPassCfg::GetInstance()
             ->IsEnabledWeightPermutePass()) {
      h_weight_tensor_hwck = h_weight_tensor.permute({2, 3, 1, 0}).contiguous();
    }
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
    // out = add(pool_out, x)
    torch::Tensor bias3 =
        torch::randn(1, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor h_bias3 = bias3.to(torch::kHPU);
    auto h_out_add = torch::add(h_pool_out, h_bias3);
    auto out_add = torch::add(pool_out, bias3);
    torch::Tensor out_add_hpu = h_out_add.to(torch::kCPU);
    EXPECT_EQ(allclose(out_add_hpu, out_add, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Graph : Conv2DTranspose Op with Bias is lowered to 3 sub kernels
//         i.e. Conv2D, Reshape and Add
//
//                           Data
//                            |
//            (weights) -> Conv2D
//                            |
//                Bias ->   Reshape
//                            |
//                           Add
//                            |
//                           Out
TEST_F(LazyDynamicComputeOutputShapesTest, Conv2DTransposeBiasTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  int kH = 3;
  int kW = 3;
  const int C = 16;
  const int N = 16;
  int H = 16;

  std::vector<int> in_sizes{16, 32, 64};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    int W = in_sizes[i];

    torch::Tensor bias = torch::randn({C}, torch::dtype(torch::kFloat));
    torch::Tensor h_bias = bias.to(torch::kHPU);
    torch::Tensor weight_tensor =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    torch::Tensor h_weight_tensor = weight_tensor.to(torch::kHPU);
    torch::Tensor in_tensor =
        torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor h_in_tensor = in_tensor.to(torch::kHPU);
    torch::Tensor h_weight_tensor_hwck = h_weight_tensor;
    if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) &&
        !habana_lazy::exec::OptPassCfg::GetInstance()
             ->IsEnabledWeightPermutePass()) {
      h_weight_tensor_hwck = h_weight_tensor.permute({2, 3, 1, 0}).contiguous();
    }
    torch::Tensor h_out_conv = torch::conv_transpose2d(
        h_in_tensor, h_weight_tensor_hwck, h_bias, 1, 0, 0, 1, 1);
    torch::Tensor out_conv =
        torch::conv_transpose2d(in_tensor, weight_tensor, bias, 1, 0, 0, 1, 1);

    torch::Tensor out_conv_hpu = h_out_conv.to(torch::kCPU);
    EXPECT_EQ(allclose(out_conv_hpu, out_conv, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyDynamicComputeOutputShapesTest, Fill) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  torch::Tensor A = torch::randn({20});
  torch::Tensor hA = A.to(torch::kHPU);
  auto hout = hA.fill_(1.0);
  auto out = hout.to(torch::kCPU);

  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

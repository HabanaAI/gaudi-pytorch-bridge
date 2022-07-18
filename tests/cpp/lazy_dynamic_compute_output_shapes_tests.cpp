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

// Test Add Add Div Sub Cat Relu
TEST_F(LazyDynamicComputeOutputShapesTest, DISABLED_AddAddDivSubCatRelu) {
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
    torch::Tensor E = torch::randn(dimentions);
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hC = C.to(torch::kHPU);
    torch::Tensor hD = D.to(torch::kHPU);
    torch::Tensor hE = E.to(torch::kHPU);
    torch::Tensor add_out1 = torch::add(hA, hB, 2.3);
    torch::Tensor add_out2 = torch::add(hC, add_out1, 3.4);
    torch::Tensor div_out3 = torch::div(add_out2, 6);
    torch::Tensor sub_out4 = torch::sub(div_out3, hD);
    torch::Tensor cat_out5 = torch::cat({sub_out4, hE}, 3);
    torch::Tensor out = torch::relu(cat_out5);

    torch::Tensor add_out1_cpu = torch::add(A, B, 2.3);
    torch::Tensor add_out2_cpu = torch::add(C, add_out1_cpu, 3.4);
    torch::Tensor div_out3_cpu = torch::div(add_out2_cpu, 6);
    torch::Tensor sub_out4_cpu = torch::sub(div_out3_cpu, D);
    torch::Tensor cat_out5_cpu = torch::cat({sub_out4_cpu, E}, 3);
    torch::Tensor out_cpu = torch::relu(cat_out5_cpu);

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

TEST_F(LazyDynamicComputeOutputShapesTest, SiluBwdTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  const int C = 16;
  const int N = 16;
  int H = 16;

  std::vector<int> in_sizes{16, 32, 64};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    int W = in_sizes[i];

    auto input_tensor = torch::randn({N, C, H, W}, torch::requires_grad(false));
    auto grad = torch::randn({N, C, H, W}, torch::requires_grad(false));

    auto hinput = input_tensor.to(torch::kHPU);
    auto hgrad = grad.to(torch::kHPU);

    auto hresult = torch::silu_backward(hgrad, hinput);
    auto hout = hresult.to(torch::kCPU);

    auto cpu_out = torch::silu_backward(grad, input_tensor);

    EXPECT_EQ(allclose(hout, cpu_out, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyDynamicComputeOutputShapesTest, UpsampleNearest2DTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  int count = -1;
  auto upsample_test = [&count](c10::IntArrayRef in_sizes) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", ++count, " ----\n");
    torch::Tensor tensor = torch::randn(in_sizes, torch::requires_grad(false));
    torch::Tensor tHabana = tensor.to(torch::kHPU);
    std::array<double, 2> scale_array = {2.0, 2.0};
    c10::ArrayRef<double> scale_factors = scale_array;
    auto outHabana = torch::upsample_nearest2d(tHabana, {}, scale_factors);
    auto out = torch::upsample_nearest2d(tensor, {}, scale_factors);
    bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
    EXPECT_EQ(equal, true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", count, " ----\n");
  };
  upsample_test({1, 1, 2, 3});
  upsample_test({1, 1, 4, 7});
  upsample_test({1, 1, 6, 12});

  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyDynamicComputeOutputShapesTest, UpsampleNearest2DBwdTest) {
  torch::manual_seed(0);
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  int count = -1;
  auto upsample_test = [&count](c10::IntArrayRef in_sizes) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", ++count, " ----\n");
    auto mat1 = torch::randn(in_sizes);
    auto mat1_h = mat1.to(torch::kHPU);
    mat1.set_requires_grad(true);
    std::array<double, 2> scales = {2.0, 3.0};
    c10::optional<c10::ArrayRef<double>> scale_factors = scales;
    c10::optional<c10::IntArrayRef> out_size = c10::nullopt;

    auto out = torch::upsample_nearest2d(mat1, out_size, scale_factors);
    auto grad_out = torch::ones_like(out);
    auto grad_out_h = grad_out.to(torch::kHPU);
    out.backward(grad_out);
    auto grad_mat1 = mat1.grad();

    torch::Tensor grad_mat1_h;

    grad_mat1_h = upsample_nearest2d_backward_hpu_lazy(
        grad_out_h, out_size, in_sizes, scale_factors);
    bool equal1 = grad_mat1.allclose(grad_mat1_h.to(torch::kCPU), 0.01, 0.01);
    EXPECT_EQ(equal1, true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", count, " ----\n");
  };
  upsample_test({1, 1, 2, 3});
  upsample_test({1, 1, 4, 7});
  upsample_test({1, 1, 6, 12});

  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyDynamicComputeOutputShapesTest, SqueezeTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  const int C = 3;
  const int N = 16;
  int H = 16;

  std::vector<int> in_sizes{16, 32, 64};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    int W = in_sizes[i];

    auto x = torch::randn({N, C, H, W}, torch::requires_grad(false));
    auto hx = x.to(torch::kHPU);

    auto B = torch::squeeze(x);
    auto hB = torch::squeeze(hx);

    EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }

  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyDynamicComputeOutputShapesTest, AllReduceStridedInsertTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }

  std::vector<int> in_sizes{16, 24, 32};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    torch::Tensor A = torch::randn({in_sizes[i]}, torch::requires_grad(false));
    auto v1 = A.view(-1);
    auto v2 = A.view(-1);
    auto grad1 = torch::randn({in_sizes[i]}, torch::requires_grad(false));
    auto grad2 = torch::randn({in_sizes[i]}, torch::requires_grad(false));

    auto hA = A.to(torch::kHPU);
    auto hv1 = hA.view(-1);
    auto hv2 = hA.view(-1);
    auto hgrad1 = grad1.to(torch::kHPU);
    auto hgrad2 = grad2.to(torch::kHPU);

    v1.add_(grad1);
    v2.add_(grad2);

    hv1.add_(hgrad1);
    hv2.add_(hgrad2);

    EXPECT_EQ(allclose(A, hA.cpu(), 0.001, 0.001), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }

  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

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
#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include "habana_kernels/lazy_kernels_declarations.h"

#include "backend/helpers/dynamic_bucket_info.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

using namespace habana_lazy;

// In this class both the pass fallback and compilation fallback are disabled
class LazyDynamicInferOutputMetasTest
    : public habana_lazy_test::LazyDynamicTest {
  void SetUp() override {
    if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
      SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
    }
    habana_lazy_test::LazyDynamicTest::SetUp();
  }

  void TearDown() override {
    UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
    habana_lazy_test::LazyDynamicTest::TearDown();
  }
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
//                            |              Broadcast
//                            |                  |
//                           Add <----------------
//                            |
//                           Out

TEST_F(LazyDynamicInferOutputMetasTest, AddConv2DBNMaxPoolTest) {
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
TEST_F(LazyDynamicInferOutputMetasTest, Conv2DTransposeBiasTest) {
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
    torch::Tensor h_out_conv = torch::conv_transpose2d(
        h_in_tensor, h_weight_tensor_hwck, h_bias, 1, 0, 0, 1, 1);
    torch::Tensor out_conv =
        torch::conv_transpose2d(in_tensor, weight_tensor, bias, 1, 0, 0, 1, 1);

    torch::Tensor out_conv_hpu = h_out_conv.to(torch::kCPU);
    EXPECT_EQ(allclose(out_conv_hpu, out_conv, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
}

TEST_F(LazyDynamicInferOutputMetasTest, Fill) {
  torch::Tensor A = torch::randn({20});
  torch::Tensor hA = A.to(torch::kHPU);
  auto hout = hA.fill_(1.0);
  auto out = hout.to(torch::kCPU);
}

TEST_F(LazyDynamicInferOutputMetasTest, SiluBwdTest) {
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
}

TEST_F(LazyDynamicInferOutputMetasTest, UpsampleNearest2DTest) {
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
}

TEST_F(LazyDynamicInferOutputMetasTest, UpsampleNearest2DBwdTest) {
  torch::manual_seed(0);
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

    std::array<int64_t, 2> out_sizes_arr = {8, 21};
    c10::IntArrayRef out_sizes = out_sizes_arr;
    c10::optional<double> scales_h(2.0);
    c10::optional<double> scales_w(3.0);
    grad_mat1_h = torch::upsample_nearest2d_backward(
        grad_out_h, out_sizes, in_sizes, scales_h, scales_w);

    bool equal1 = grad_mat1.allclose(grad_mat1_h.to(torch::kCPU), 0.01, 0.01);
    EXPECT_EQ(equal1, true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", count, " ----\n");
  };
  upsample_test({1, 1, 2, 3});
  upsample_test({1, 1, 4, 7});
  upsample_test({1, 1, 6, 12});
}

TEST_F(LazyDynamicInferOutputMetasTest, SqueezeTest) {
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
}

TEST_F(LazyDynamicInferOutputMetasTest, AllReduceStridedInsertTest) {
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
}

TEST_F(LazyDynamicInferOutputMetasTest, AllReduceStridedViewTest) {
  std::vector<int> in_sizes{16, 24, 32};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    torch::Tensor A = torch::randn({in_sizes[i]}, torch::requires_grad(false));
    auto v1 = A.as_strided({in_sizes[i]}, {1}, 0);
    auto v2 = A.as_strided({in_sizes[i]}, {1}, 0);
    auto grad1 = torch::randn({in_sizes[i]}, torch::requires_grad(false));
    auto grad2 = torch::randn({in_sizes[i]}, torch::requires_grad(false));

    auto hA = A.to(torch::kHPU);
    auto hv1 = hA.as_strided({in_sizes[i]}, {1}, 0);
    auto hv2 = hA.as_strided({in_sizes[i]}, {1}, 0);
    auto hgrad1 = grad1.to(torch::kHPU);
    auto hgrad2 = grad2.to(torch::kHPU);

    v1.add_(grad1);
    v2.add_(grad2);

    hv1.add_(hgrad1);
    hv2.add_(hgrad2);

    EXPECT_EQ(allclose(A, hA.cpu(), 0.001, 0.001), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
}

TEST_F(LazyDynamicInferOutputMetasTest, EqScalarTest) {
  torch::Tensor A = torch::rand({2, 2}, torch::requires_grad(false));
  float compVal = 1.1f;
  auto out_cpu = torch::eq(A, compVal);

  auto hA = A.to(torch::kHPU);
  auto result = torch::eq(hA, compVal);
  torch::Tensor out_hpu = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(out_cpu.to(torch::kFloat), out_hpu.to(torch::kFloat)), true);
}

TEST_F(LazyDynamicInferOutputMetasTest, ge) {
  auto cpu_in1 = torch::randn({42}).to(at::kBFloat16);
  auto cpu_in2 = torch::randn({2, 42});

  auto hpu_in1 = cpu_in1.to("hpu");
  auto hpu_in2 = cpu_in2.to("hpu");

  EXPECT_TRUE(
      at::allclose(at::ge(cpu_in1, cpu_in2), at::ge(hpu_in1, hpu_in2).cpu()));
}

TEST_F(LazyDynamicInferOutputMetasTest, Maximum) {
  torch::Tensor input1 = torch::randn({2, 2});
  torch::Tensor input2 = torch::randn({2, 2});

  torch::Tensor out_cpu = at::max(input1, input2);
  torch::Tensor out_hpu =
      at::max(input1.to(torch::kHPU), input2.to(torch::kHPU));
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyDynamicInferOutputMetasTest, Minimum) {
  torch::Tensor input1 = torch::randn({2, 2});
  torch::Tensor input2 = torch::randn({2, 2});

  torch::Tensor out_cpu = at::min(input1, input2);
  torch::Tensor out_hpu =
      at::min(input1.to(torch::kHPU), input2.to(torch::kHPU));
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyDynamicInferOutputMetasTest, upsample_bicubic2d_fwd_scale) {
  torch::Tensor input1 = torch::randn({2, 7, 3, 4});
  torch::Tensor input1hpu = input1.to("hpu");
  std::vector<double> scale_factor = {1.999, 2.999};

  auto expected = torch::upsample_bicubic2d(
      input1,
      c10::nullopt,
      /*align_corner*/ false,
      scale_factor);
  auto result = torch::upsample_bicubic2d(
      input1hpu,
      c10::nullopt,
      /*align_corner*/ false,
      scale_factor);

  EXPECT_EQ(allclose(expected, result.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyDynamicInferOutputMetasTest, upsample_bicubic2d_bwd_size) {
  torch::Tensor input1 = torch::randn({4, 3, 12, 64});
  torch::Tensor input1hpu = input1.to("hpu");
  std::vector<int64_t> output_size = {12, 64};
  std::vector<int64_t> input_size = {4, 3, 6, 32};

  auto expected = torch::upsample_bicubic2d_backward(
      input1, output_size, input_size, /*align_corner*/ true);
  auto result = torch::upsample_bicubic2d_backward(
      input1hpu, output_size, input_size, /*align_corner*/ true);

  EXPECT_EQ(allclose(expected, result.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyDynamicInferOutputMetasTest, sigmoid) {
  torch::Tensor cpu_in = torch::randn({4, 3, 12, 64});
  torch::Tensor hpu_in = cpu_in.to("hpu");

  EXPECT_TRUE(at::allclose(at::sigmoid(cpu_in), at::sigmoid(hpu_in).cpu()));
}

TEST_F(LazyDynamicInferOutputMetasTest, SigmoidBwdTest) {
  auto input_tensor =
      torch::arange(4, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 2, 2});
  auto grad_tensor =
      torch::arange(4, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 2, 2});
  torch::Tensor cpu_out = torch::sigmoid_backward(grad_tensor, input_tensor);

  torch::Tensor tHabanaI = input_tensor.to(torch::kHPU);
  torch::Tensor tHabanaG = grad_tensor.to(torch::kHPU);
  torch::Tensor hout_backward = torch::sigmoid_backward(tHabanaG, tHabanaI);
  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(hout_backward)};
  HbLazyTensor::SyncTensorsGraph(&tensors);
  auto hout_lazy = hout_backward.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out), true);
}

// index_select is supported as manual op
TEST_F(LazyDynamicInferOutputMetasTest, index_select) {
  auto in_size = 1;
  auto max_value = 1024;
  auto datatype = torch::kInt;
  auto index_value = 4;
  auto dim = 0;
  auto out_size = 0;

  torch::ScalarType dtype = datatype;
  torch::Tensor cpu_in =
      torch::randint(0, max_value, {index_value}, torch::kInt);
  torch::Tensor hpu_in = cpu_in.to("hpu");
  auto cpu_index = cpu_in.to(torch::kLong);
  auto hpu_index = cpu_index.to(torch::kHPU);

  torch::Tensor cpu_in_1 = torch::randint(-127, 128, {max_value}, torch::kInt);
  torch::Tensor hpu_in_2 = cpu_in_1.to("hpu");

  auto expected = torch::index_select(cpu_in_1, dim, cpu_index);
  auto result = torch::index_select(hpu_in_2, dim, hpu_index);

  EXPECT_EQ(allclose(expected, result.cpu(), 0, 0), true);
}

TEST_F(LazyDynamicInferOutputMetasTest, index_select_1) {
  auto max_value = 28;
  auto datatype = torch::kInt;
  auto index_value = 5;
  auto dim = 1;
  auto out_size = 0;

  torch::ScalarType dtype = datatype;
  torch::Tensor cpu_in =
      torch::randint(0, max_value, {index_value}, torch::kInt);
  torch::Tensor hpu_in = cpu_in.to("hpu");
  auto cpu_index = cpu_in.to(torch::kLong);
  auto hpu_index = cpu_index.to(torch::kHPU);
  auto expected = torch::empty(out_size, dtype);
  auto result =
      torch::empty(out_size, torch::TensorOptions(dtype).device("hpu"));

  torch::Tensor cpu_in_1 = torch::randint(-127, 128, {28, 28}, torch::kInt);
  torch::Tensor hpu_in_2 = cpu_in_1.to("hpu");

  torch::index_select_outf(cpu_in_1, dim, cpu_index, expected);
  torch::index_select_outf(hpu_in_2, dim, hpu_index, result);

  EXPECT_EQ(allclose(expected, result.cpu(), 0, 0), true);
}

// index_select_out is supported as auto gen op
TEST_F(LazyDynamicInferOutputMetasTest, index_select_out) {
  auto in_size = 1;
  auto max_value = 1024;
  auto datatype = torch::kInt;
  auto index_value = 4;
  auto dim = 0;
  auto out_size = 0;

  torch::ScalarType dtype = datatype;
  torch::Tensor cpu_in =
      torch::randint(0, max_value, {index_value}, torch::kInt);
  torch::Tensor hpu_in = cpu_in.to("hpu");
  auto cpu_index = cpu_in.to(torch::kLong);
  auto hpu_index = cpu_index.to(torch::kHPU);
  auto expected = torch::empty(out_size, dtype);
  auto result =
      torch::empty(out_size, torch::TensorOptions(dtype).device("hpu"));

  torch::Tensor cpu_in_1 = torch::randint(-127, 128, {max_value}, torch::kInt);
  torch::Tensor hpu_in_2 = cpu_in_1.to("hpu");

  torch::index_select_outf(cpu_in_1, dim, cpu_index, expected);
  torch::index_select_outf(hpu_in_2, dim, hpu_index, result);

  EXPECT_EQ(allclose(expected, result.cpu(), 0, 0), true);
}

TEST_F(LazyDynamicInferOutputMetasTest, index_select_out_1) {
  auto max_value = 28;
  auto datatype = torch::kInt;
  auto index_value = 5;
  auto dim = 1;
  auto out_size = 0;

  torch::ScalarType dtype = datatype;
  torch::Tensor cpu_in =
      torch::randint(0, max_value, {index_value}, torch::kInt);
  torch::Tensor hpu_in = cpu_in.to("hpu");
  auto cpu_index = cpu_in.to(torch::kLong);
  auto hpu_index = cpu_index.to(torch::kHPU);
  auto expected = torch::empty(out_size, dtype);
  auto result =
      torch::empty(out_size, torch::TensorOptions(dtype).device("hpu"));

  torch::Tensor cpu_in_1 = torch::randint(-127, 128, {28, 28}, torch::kInt);
  torch::Tensor hpu_in_2 = cpu_in_1.to("hpu");

  torch::index_select_outf(cpu_in_1, dim, cpu_index, expected);
  torch::index_select_outf(hpu_in_2, dim, hpu_index, result);

  EXPECT_EQ(allclose(expected, result.cpu(), 0, 0), true);
}

// Also validates InferOutputMeta for for Reshape/View
TEST_F(LazyDynamicInferOutputMetasTest, ReshapeTest) {
  const int N = 2;
  const int C = 4;
  const int H = 8;
  std::vector<int> in_sizes{16, 32, 64};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");

    int W = in_sizes[i];
    auto A = torch::randn({N * C * H * W});
    auto A_reshape = A.reshape({N, C, H, W});

    auto hA = A.to(torch::kHPU);
    auto hA_reshape = hA.reshape({N, C, H, W});

    EXPECT_TRUE(allclose(hA_reshape.to(torch::kCPU), A_reshape));
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
}

TEST_F(LazyDynamicInferOutputMetasTest, squeezeCmptOpTest) {
  auto x = torch::randn({4});
  auto hx = x.to(torch::kHPU);

  auto B = torch::squeeze(x);
  auto hB = torch::squeeze(hx);

  EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyDynamicInferOutputMetasTest, SliceTest_CmptOtShp) {
  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);
  int64_t dim = 1;
  int64_t start_index = 0;
  int64_t end = 8;
  int64_t step = 1;

  auto h_out = torch::slice(h_a, dim, start_index, end, step);

  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::slice(a, dim, start_index, end, step);

  EXPECT_EQ(allclose(h_cout, cout), true);
}

// Also validates InferOutputMeta for View, AddInplace and strided_insert
TEST_F(LazyDynamicInferOutputMetasTest, AddInplaceViewTest) {
  int N = 1;
  int C = 2;
  int H = 4;
  at::Scalar alpha = 0.5;
  at::Scalar Y = 2.0;
  std::vector<int> in_sizes{8, 10, 12, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor C = A.view(-1);
    torch::Tensor out_cpu = C.add_(alpha);
    torch::Tensor hC = hA.view(-1);
    torch::Tensor out_hpu = hC.add_(alpha);
    auto out = out_hpu.to(torch::kCPU);
    EXPECT_EQ(allclose(out, out_cpu, 0.001, 0.001), true);
  }
}

// Also validates InferOutputMeta for ArangeHtF32
TEST_F(LazyDynamicInferOutputMetasTest, ArangeTestFloatHt) {
  std::vector<int> start_sizes{0, 0, 0, 0, 0};
  std::vector<int> end_sizes{5, 10, 15, 20, 25};
  std::vector<int> step_sizes{1, 2, 3, 4, 5};
  for (int i = 0; i < start_sizes.size(); i++) {
    SET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR, true, 1);
    torch::Scalar start = start_sizes[i];
    torch::Scalar end = end_sizes[i];
    torch::Scalar step = step_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");

    c10::optional<at::ScalarType> dtype = c10::ScalarType::Float;

    c10::optional<at::Device> hb_device = at::DeviceType::HPU;
    at::TensorOptions hb_options =
        at::TensorOptions().dtype(dtype).device(hb_device);
    c10::optional<at::Device> cpu_device = at::DeviceType::CPU;
    at::TensorOptions cpu_options =
        at::TensorOptions().dtype(dtype).device(cpu_device);

    auto h_a = torch::arange(start, end, step, hb_options);
    auto h_cout = h_a.to(torch::kCPU);
    auto a = torch::arange(start, end, step, cpu_options);
    UNSET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR);
    EXPECT_EQ(allclose(h_cout, a), true);
  }
}

// Also validates InferOutputMeta for ArangeHtI32
TEST_F(LazyDynamicInferOutputMetasTest, ArangeTestHt) {
  // std::vector<int> start_sizes{1, 1, 1, 1};
  std::vector<int> start_sizes{0, 0, 0, 0, 0};
  std::vector<int> end_sizes{5, 10, 15, 20, 25};
  std::vector<int> step_sizes{1, 2, 3, 4, 5};
  for (int i = 0; i < start_sizes.size(); i++) {
    SET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR, true, 1);
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
    UNSET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR);
    EXPECT_EQ(allclose(h_cout, a), true);
  }
}

TEST_F(LazyDynamicInferOutputMetasTest, DISABLED_RoiAlignBwd) {
  auto roi_align_test = [](int num_boxes, std::vector<int64_t> input_shape) {
    auto images = torch::randn(input_shape).to(torch::kHPU);
    auto boxes = torch::randn({num_boxes, 4}) * 64;
    // ensure x2 > x1 and y2 > y1
    auto tlist = boxes.split(2, 1);
    tlist[1] = tlist[1] + tlist[0];
    auto new_boxes = torch::cat({tlist[0], tlist[1]}, 1).to(torch::kHPU);
    auto num_rois =
        torch::randint(0, 2, {num_boxes}, torch::kInt).to(torch::kHPU);
    new_boxes.set_requires_grad(true);
    auto output = roi_align_fwd_hpu_lazy(
        images, new_boxes, num_rois, 7, 7, 0, 2, 0.25, true);
    auto sizes = images.sizes();
    auto temp = roi_align_bwd_hpu_lazy(
        output,
        new_boxes,
        num_rois,
        sizes[0],
        sizes[1],
        sizes[3],
        sizes[4],
        2,
        0.25,
        true);
    output.to(torch::kCPU);
  };
  roi_align_test(6, {2, 3, 25, 25});
  roi_align_test(10, {2, 3, 35, 35});
  roi_align_test(12, {2, 3, 50, 50});
}

// Also validates InferOutputMeta for RandPermHT
TEST_F(LazyDynamicInferOutputMetasTest, RandPermHT) {
  SET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR, true, 1);
  SET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_RANDPERM_HOST_TENSOR, true, 1);
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
  UNSET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_RANDPERM_HOST_TENSOR);
  UNSET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR);
}

TEST_F(LazyDynamicInferOutputMetasTest, Mean) {
  std::vector<int> in_sizes{16, 24, 32};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    torch::Tensor A = torch::randn({in_sizes[i]}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hOut = torch::mean(hA);
    torch::Tensor Out = torch::mean(A);
    EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out));
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
}

void repeatInlvTest(
    at::Tensor A,
    std::vector<int64_t> rpt_vals,
    int64_t dim = -1);

TEST_F(LazyDynamicInferOutputMetasTest, repeatInlv) {
  SET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR, true, 1);
  repeatInlvTest(torch::tensor({4, 5}), {10, 7});
  repeatInlvTest(torch::randn({4, 5}), {2, 1, 1, 1, 1}, 1);
  UNSET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR);
}

TEST_F(LazyDynamicInferOutputMetasTest, RepeatTest) {
  SET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR, true, 1);
  int H = 4;
  std::vector<int> c{5, 50, 100};
  std::vector<int> in_sizes{10, 231, 520};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);

    torch::Tensor h_out = hA.repeat({c[i], 1, 1});

    auto h_cout = h_out.to(torch::kCPU);
    auto cout = A.repeat({c[i], 1, 1});
    EXPECT_EQ(allclose(h_cout, cout), true);
  }
  UNSET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR);
}

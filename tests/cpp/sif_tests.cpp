/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;
using namespace at;

class SifTest : public habana_lazy_test::LazyDynamicTest {
 protected:
  void validate_shape_start() {
    if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
      SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }

  void validate_shape_end() {
    UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
  }

  void validate_sif_start() {
    if (false == GET_ENV_FLAG_NEW(PT_HPU_RUN_HYBRID_SIF))
      SET_ENV_FLAG_NEW(PT_HPU_RUN_HYBRID_SIF, true, 1);
    if (false == GET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE))
      SET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE, true, 1);
  }

  void validate_sif_end() {
    UNSET_ENV_FLAG_NEW(PT_HPU_RUN_HYBRID_SIF);
    UNSET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE);
  }
};

TEST_F(SifTest, Slice) {
  int N = 1;
  int C = 4;
  int H = 24;
  std::vector<int> W_values{16, 36};
  std::vector<int> rounds{1, 2};
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

TEST_F(SifTest, SimpleGraph) {
  validate_shape_start();
  int kH = 3;
  int kW = 3;
  const int C = 16;
  const int N = 16;
  int H = 16;

  std::vector<int> in_sizes{16, 24, 32};
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
        in_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, {1});
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
  validate_shape_end();
}

// Test Add Add Div Sub
TEST_F(SifTest, AddAddDivSub) {
  validate_shape_start();
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
  validate_shape_end();
}

// Test Add Add Div Sub Cat Relu
TEST_F(SifTest, AddAddDivSubCatRelu) {
  validate_shape_start();
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
  validate_shape_end();
}

TEST_F(SifTest, SingleOpCat) {
  PT_TEST_DEBUG("SingleOpCat_BEGIN");
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));

  auto exp = torch::cat({A, B});

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);

  torch::Tensor out = torch::cat({hA, hB});
  auto result = out.to(torch::kCPU);
  PRINT_TENSOR(out);
  EXPECT_EQ(allclose(result, exp), true);
  PT_TEST_DEBUG("SingleOpCat_END");
}

TEST_F(SifTest, AddMulRelu) {
  validate_shape_start();
  int A = 50;
  const int C = 30;

  std::vector<int> input_sizes{34, 16, 24};
  std::vector<int> test_rounds{1, 1, 1};

  for (int i = 0; i < input_sizes.size(); i++) {
    for (int j = 1; j <= test_rounds[i]; j++) {
      int B = input_sizes[i];
      PT_TEST_DEBUG("PTI_DBG :: TEST ", "  START");

      torch::Tensor h0 =
          torch::randn({C, B, A}, torch::requires_grad(false)).to(torch::kHPU);
      torch::Tensor h1 =
          torch::randn({C, B, A}, torch::requires_grad(false)).to(torch::kHPU);

      torch::Tensor h4 = torch::add(h0, h1);
      torch::Tensor h5 = torch::mul(h0, h1);
      torch::Tensor h6 = torch::mul(h4, h5);
      torch::Tensor h7 = torch::relu(h6);
      HbLazyTensor::StepMarker({});

      // auto h7_c = h7.to(torch::kCPU);
      // PRINT_TENSOR(h7_c);

      PT_TEST_DEBUG("PTI_DBG :: TEST ", "  END");
    }
  }
  validate_shape_end();
}

// Graph : Cat + Reshape + Relu + Conv2DTransposeBias
// 1. Cat Op
// 2. Reshape Op
// 3. Relu Op, HPU Op
// 4. Conv2DTranspose Compound Op with Bias is lowered to 3 sub kernels
//         i.e. Conv2D, Reshape and Add
//                           Data
//                            |
//            (weights) -> Conv2D (adds Intermediate Shape Tensor1)
//                            |
//                Bias ->   Reshape (adds Intermediate Shape Tensor2)
//                            |
//                           Add
//                            |
//                           Out
//
TEST_F(SifTest, Cat_Reshape_Relu_Conv2DTransposeBias_Test) {
  validate_shape_start();
  int kH = 3;
  int kW = 3;
  const int C = 16;
  const int N = 16;
  int H = 16;

  std::vector<int> in_sizes{16, 32, 64};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    int W = in_sizes[i];
    // 1. Cat Node
    auto tensor_1 =
        torch::randn({N * C * H * (W / 2)}, torch::requires_grad(false));
    auto tensor_2 =
        torch::randn({N * C * H * (W / 2)}, torch::requires_grad(false));
    auto cat_tensor = torch::cat({tensor_1, tensor_2}, 0);

    auto h_tensor_1 = tensor_1.to(torch::kHPU);
    auto h_tensor_2 = tensor_2.to(torch::kHPU);
    auto h_cat_tensor = torch::cat({h_tensor_1, h_tensor_2}, 0);

    // 2. Reshape Node
    auto reshape_tensor = cat_tensor.reshape({N, C, H, W});
    auto h_reshape_tensor = h_cat_tensor.reshape({N, C, H, W});

    // 3. Relu Node
    auto relu_tensor = torch::relu(reshape_tensor);
    auto h_relu_tensor = torch::relu(h_reshape_tensor);

    // 4. ConvTranpsoseBias Node => Compound Op (Conv + Reshape + Add)
    auto bias = torch::randn({C}, torch::dtype(torch::kFloat));
    auto h_bias = bias.to(torch::kHPU);
    auto weight_tensor =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    auto h_weight_tensor = weight_tensor.to(torch::kHPU);
    auto h_weight_tensor_hwck = h_weight_tensor;
    auto h_out_conv = torch::conv_transpose2d(
        h_relu_tensor, h_weight_tensor_hwck, h_bias, 1, 0, 0, 1, 1);
    auto out_conv = torch::conv_transpose2d(
        relu_tensor, weight_tensor, bias, 1, 0, 0, 1, 1);

    auto out_conv_hpu = h_out_conv.to(torch::kCPU);
    EXPECT_EQ(allclose(out_conv_hpu, out_conv, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
  validate_shape_end();
}

TEST_F(SifTest, AllReduce_Test) {
  validate_shape_start();

  std::vector<int> in_sizes{16, 24, 32};
  for (int i = 0; i < in_sizes.size(); i++) {
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

    v1.mul_(grad1);
    v2.mul_(grad2);

    hv1.mul_(hgrad1);
    hv2.mul_(hgrad2);

    EXPECT_EQ(allclose(A, hA.cpu(), 0.001, 0.001), true);
  }
  validate_shape_end();
}

TEST_F(SifTest, AllReduceWithControlEdge_Test) {
  validate_shape_start();

  std::vector<int> in_sizes{16, 24, 32};
  for (int i = 0; i < in_sizes.size(); i++) {
    torch::Tensor A = torch::randn({in_sizes[i]}, torch::requires_grad(false));
    auto b = torch::relu(A);
    auto v1 = A.view(-1);
    auto v2 = A.view(-1);
    auto grad1 = torch::randn({in_sizes[i]}, torch::requires_grad(false));
    auto grad2 = torch::randn({in_sizes[i]}, torch::requires_grad(false));

    auto hA = A.to(torch::kHPU);
    auto hB = torch::relu(hA);
    auto hv1 = hA.view(-1);
    auto hv2 = hA.view(-1);
    auto hgrad1 = grad1.to(torch::kHPU);
    auto hgrad2 = grad2.to(torch::kHPU);

    v1.mul_(grad1);
    v2.mul_(grad2);

    hv1.mul_(hgrad1);
    hv2.mul_(hgrad2);

    HbLazyTensor::StepMarker({});

    EXPECT_EQ(allclose(A, hA.cpu(), 0.001, 0.001), true);
  }
  validate_shape_end();
}

// To Do: Enable this below test once InferOutputMeta is fixed for at::prod
TEST_F(SifTest, DISABLED_Fill_Add) {
  validate_shape_start();

  std::vector<int> in_sizes{16, 32, 64};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    auto tensor = torch::randn({in_sizes[i]});
    auto h_tensor = tensor.to(torch::kHPU);

    auto tensor_fill = tensor.fill_(1.0);
    auto h_tensor_fill = h_tensor.fill_(1.0);

    auto tensor_2 = torch::randn({in_sizes[i]});
    auto h_tensor_2 = tensor_2.to(torch::kHPU);
    auto out = torch::add(tensor_fill, tensor_2);
    auto h_out = torch::add(h_tensor_fill, h_tensor_2);

    EXPECT_EQ(allclose(out, h_out, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
  validate_shape_end();
}

// Hybrid SIF test, Tests Index, Sub - auto gen ops, Cat manual op
TEST_F(SifTest, IndexSubCat) {
  validate_shape_start();
  validate_sif_start();
  std::vector<int> in_sizes{8, 16, 32};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    int H = in_sizes[i];
    int W = in_sizes[i];
    auto a = torch::randn(
        {H, W}, torch::dtype(torch::kBFloat16).requires_grad(false));
    auto h_a = a.to(torch::kHPU);

    auto idx = torch::randint(0, W - 1, {W}, torch::dtype(torch::kInt64));
    auto h_idx = idx.to(torch::kHPU);

    auto idx2 = torch::randint(0, W - 1, {W}, torch::dtype(torch::kInt64));
    auto h_idx2 = idx2.to(torch::kHPU);

    // Make list
    c10::List<c10::optional<at::Tensor>> indices_cpu_list;
    c10::List<c10::optional<at::Tensor>> indices_hpu_list;

    indices_cpu_list.push_back(idx);
    indices_cpu_list.push_back(idx2);
    indices_hpu_list.push_back(h_idx);
    indices_hpu_list.push_back(h_idx2);

    auto index_out = torch::index(a, indices_cpu_list);
    auto h_index_out = torch::index(h_a, indices_hpu_list);

    auto b = torch::randn({W}, torch::requires_grad(false));
    auto h_b = b.to(torch::kHPU);

    auto sub_out = torch::sub(index_out, b, 1);
    auto h_sub_out = torch::sub(h_index_out, h_b, 1);

    auto c = torch::randn({W}, torch::requires_grad(false));
    auto h_c = c.to(torch::kHPU);

    auto out_cpu = torch::cat({sub_out, c}, 0);
    auto out_hpu = torch::cat({h_sub_out, h_c}, 0);

    EXPECT_EQ(
        allclose(
            out_hpu.to(torch::kFloat).to(torch::kCPU),
            out_cpu.to(torch::kFloat),
            0.01,
            0.01),
        true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
  validate_shape_end();
  validate_sif_end();
}

// Hybrid SIF test, Tests Index, Sub and Silu Bwd - auto gen ops
// with enabled Compute Output shape.
TEST_F(SifTest, IndexSubSiluBwd) {
  validate_shape_start();
  validate_sif_start();
  std::vector<int> in_sizes{8, 16, 32};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    int H = in_sizes[i];
    int W = in_sizes[i];
    auto a =
        torch::randn({H, W}, torch::dtype(torch::kFloat).requires_grad(false));
    auto h_a = a.to(torch::kHPU);

    auto idx = torch::randint(0, W - 1, {W}, torch::dtype(torch::kInt64));
    auto h_idx = idx.to(torch::kHPU);

    auto idx2 = torch::randint(0, W - 1, {W}, torch::dtype(torch::kInt64));
    auto h_idx2 = idx2.to(torch::kHPU);

    // Make list
    c10::List<c10::optional<at::Tensor>> indices_cpu_list;
    c10::List<c10::optional<at::Tensor>> indices_hpu_list;

    indices_cpu_list.push_back(idx);
    indices_cpu_list.push_back(idx2);
    indices_hpu_list.push_back(h_idx);
    indices_hpu_list.push_back(h_idx2);

    // 1. Index Op
    auto index_out = torch::index(a, indices_cpu_list);
    auto h_index_out = torch::index(h_a, indices_hpu_list);

    auto b = torch::randn({W}, torch::requires_grad(false));
    auto h_b = b.to(torch::kHPU);

    // 2. Sub Op
    auto sub_out = torch::sub(index_out, b, 1);
    auto h_sub_out = torch::sub(h_index_out, h_b, 1);

    auto c = torch::randn({W}, torch::requires_grad(false));
    auto h_c = c.to(torch::kHPU);

    // 3. Try Silu Bwd with dummy grad tensor
    auto grad_ones = torch::ones_like(sub_out);
    auto h_grad_ones = grad_ones.to(torch::kHPU);
    auto out_cpu = torch::silu_backward(grad_ones, sub_out);
    auto out_hpu = torch::silu_backward(h_grad_ones, h_sub_out);

    EXPECT_EQ(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
  validate_shape_end();
  validate_sif_end();
}

// Hybrid SIF test, Tests Matmul/Matmul Bwd with out InferOutputMeta
// Matmul/Matmul Bwd does not support InferOutputMeta
// Skip InferOutputMeta validation
TEST_F(SifTest, MatmulFwdBwd) {
  validate_sif_start();
  std::vector<int> in_sizes{2, 4, 8};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    auto mat1 = torch::randn({2, in_sizes[i]}, torch::requires_grad());
    auto mat2 = torch::randn({in_sizes[i], 4}, torch::requires_grad());
    auto mat1_h = mat1.to(torch::kHPU);
    auto mat2_h = mat2.to(torch::kHPU);
    // retain_grad() as mat1_h and mat2_h are non-leaf tensors
    mat1_h.retain_grad();
    mat2_h.retain_grad();

    auto out = torch::matmul(mat1, mat2);
    auto grad_out = torch::ones_like(out);
    out.backward(grad_out);
    auto grad_mat1 = mat1.grad().clone().detach();
    auto grad_mat2 = mat2.grad().clone().detach();

    auto out_h = torch::matmul(mat1_h, mat2_h);
    auto grad_out_h = grad_out.to(torch::kHPU);
    out_h.backward(grad_out_h);
    auto grad_mat1_h = mat1_h.grad();
    auto grad_mat2_h = mat2_h.grad();

    EXPECT_EQ(
        allclose(grad_mat1, grad_mat1_h.to(torch::kCPU), 0.01, 0.01), true);
    EXPECT_EQ(
        allclose(grad_mat2, grad_mat2_h.to(torch::kCPU), 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
  validate_sif_end();
}

// Hybrid SIF test, Tests RandPermHT
// with enabled Compute Output shape.
TEST_F(SifTest, RandPermHT) {
  validate_sif_start();
  validate_shape_start();
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
  validate_sif_end();
  validate_shape_end();
  UNSET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_RANDPERM_HOST_TENSOR);
  UNSET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR);
}

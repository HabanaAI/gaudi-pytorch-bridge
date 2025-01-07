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

// In this class the pass fallback is enabld and compilation fallback is
// disabled
class LazyDynamicFallbackTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    SetDynamicMode();
    habana_lazy_test::LazyTest::SetUp();
  }

  void TearDown() override {
    UnsetDynamicMode();
    habana_lazy_test::LazyTest::TearDown();
  }
};

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

TEST_F(LazyDynamicFallbackTest, DynamicShapeTest4) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  const int N = 16;
  int H = 16;
  at::Scalar inScalar = 2.0;
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
}

TEST_F(LazyDynamicFallbackTest, FallbackCatTest) {
  int H = 4;
  std::vector<int> in_sizes{8, 16, 32};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({W}).to(torch::kInt32);
    torch::Tensor B = torch::randn({H}).to(torch::kInt32);
    torch::Tensor C = torch::randn({H + W}).to(torch::kInt32);
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hC = C.to(torch::kHPU);
    torch::Tensor cat_out = torch::cat({A, B});
    torch::Tensor h_cat_out = torch::cat({hA, hB});

    torch::Tensor hOut = torch::add(hC, h_cat_out);
    torch::Tensor out = torch::add(C, cat_out);
    EXPECT_EQ(allclose(hOut.to(torch::kCPU), out, 0.001, 0.001), true);
  }
}

// Also validates InferOutputMeta for concat and strided_view
TEST_F(LazyDynamicFallbackTest, MaskRcnnAsStridedTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  int H = 7;
  std::vector<int> in_sizes{100, 110, 120};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({W, 4}).to(torch::kInt32);
    torch::Tensor B = torch::randn({H, 4}).to(torch::kInt32);
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor cat_out = torch::cat({A, B});
    torch::Tensor h_cat_out = torch::cat({hA, hB});

    std::vector<int64_t> sz{W + H, 2};
    std::vector<int64_t> str{4, 1};
    c10::IntArrayRef sizes(sz.data(), sz.size());
    c10::IntArrayRef strides(str.data(), str.size());
    int64_t offset = 0;
    torch::Tensor hOut = torch::as_strided(h_cat_out, sizes, strides, offset);
    torch::Tensor out = torch::as_strided(cat_out, sizes, strides, offset);
    EXPECT_EQ(allclose(hOut.to(torch::kCPU), out, 0.001, 0.001), true);
  }
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Its places here only because its getting bucket hit and max calculation fail.
TEST_F(LazyDynamicFallbackTest, ViewTest) {
  int N = 2;
  int C = 4;
  at::Scalar alpha = 1.0;
  at::Scalar Y = 2.0;
  std::vector<int> in_sizes{6, 8, 10};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    int H = in_sizes[i] / 2;
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    std::vector<int64_t> shape{N, C, H * W, 1};
    torch::Tensor C = A.reshape(c10::IntArrayRef(shape));
    torch::Tensor hC = hA.reshape(c10::IntArrayRef(shape));
    auto C_out = hC.to(torch::kCPU);
    EXPECT_EQ(allclose(C, C_out, 0.001, 0.001), true);
  }
}

TEST_F(LazyDynamicFallbackTest, SliceTest) {
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
}

TEST_F(LazyDynamicFallbackTest, SliceTest2) {
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
}

TEST_F(LazyDynamicFallbackTest, SliceTest3) {
  int N = 24;
  std::vector<int> W_values{16, 16};
  std::vector<int> in_start{0, 1};
  std::vector<int> in_end{14, 13};
  std::vector<int> in_step{1, 1};
  for (int i = 0; i < W_values.size(); i++) {
    int W = W_values[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({N, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    int64_t dim = 1;
    int64_t start = in_start[i];
    int64_t end = in_end[i];
    int64_t step = in_step[i];
    torch::Tensor h_out = torch::slice(hA, dim, start, end, step);
    auto h_cout = h_out.to(torch::kCPU);

    auto cout = torch::slice(A, dim, start, end, step);
    EXPECT_EQ(allclose(h_cout, cout), true);
  }
}

TEST_F(LazyDynamicFallbackTest, SliceTest4) {
  int N = 128;
  std::vector<int> W_values{140, 141};
  std::vector<int> in_start{0, 1};
  std::vector<int> in_end{120, 100};
  for (int i = 0; i < W_values.size(); i++) {
    int W = W_values[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({1, N, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    int64_t dim = 2;
    int64_t start = in_start[i];
    int64_t end = in_end[i];
    hA = torch::slice(hA, 2, 0, 128, 1);
    auto h_cout = torch::slice(hA, 2, start, end, 1);
    h_cout = h_cout.to(torch::kCPU);

    A = torch::slice(A, 2, 0, 128, 1);
    auto cout = torch::slice(A, 2, start, end, 1);

    EXPECT_EQ(allclose(h_cout, cout), true);
  }
}

TEST_F(LazyDynamicFallbackTest, DynamicAvgPoolBkwdTest) {
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
}

TEST_F(LazyDynamicFallbackTest, DynamicMaxPoolBkwdTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
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
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyDynamicFallbackTest, ArangeTest) {
  // std::vector<int> start_sizes{1, 1, 1, 1};
  std::vector<int> start_sizes{0, 0, 0, 0, 0};
  std::vector<int> end_sizes{5, 10, 15, 20, 25};
  std::vector<int> step_sizes{1, 2, 3, 4, 5};
  for (int i = 0; i < start_sizes.size(); i++) {
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES, true, 1);
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
    UNSET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
    EXPECT_EQ(allclose(h_cout, a), true);
  }
}

TEST_F(LazyDynamicFallbackTest, ArangeTestFloat) {
  std::vector<int> start_sizes{0, 0, 0, 0, 0};
  std::vector<int> end_sizes{5, 10, 15, 20, 25};
  std::vector<int> step_sizes{1, 2, 3, 4, 5};
  for (int i = 0; i < start_sizes.size(); i++) {
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES, true, 1);
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
    UNSET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
    EXPECT_EQ(allclose(h_cout, a), true);
  }
}

TEST_F(LazyDynamicFallbackTest, UniqueGraph_Broadcast) {
  // unset the env variable if set for this case
  bool org_state = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_BROADCAST_BUCKET_HANDLING);
  SET_ENV_FLAG_NEW(PT_HPU_ENABLE_BROADCAST_BUCKET_HANDLING, true, 1);
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

TEST_F(LazyDynamicFallbackTest, maxpool_2d_with_indices_backward) {
  // Static Maxpool Fwd +BWD
  if (1) {
    torch::Tensor A = torch::randn(
        {2, 8, 40, 121}, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor B = torch::randn(
        {2, 8, 20, 61}, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    std::vector<int64_t> kernel_size = {{3, 3}};
    std::vector<int64_t> stride = {{2, 2}};
    std::vector<int64_t> pad_size = {{1, 1}};
    std::vector<int64_t> dilation = {{1, 1}};
    bool ceil_mode = false;

    auto maxpool_cpu = torch::max_pool2d_with_indices(
        A, kernel_size, stride, pad_size, dilation, ceil_mode);
    auto maxpool_hpu = torch::max_pool2d_with_indices(
        hA, kernel_size, stride, pad_size, dilation, ceil_mode);

    auto expected_indices = std::get<1>(maxpool_cpu);
    auto max_pool_fwd_out_0 = std::get<0>(maxpool_cpu);
    auto result_indices = std::get<1>(maxpool_hpu);

    auto expected_gradinp = torch::max_pool2d_with_indices_backward(
        B,
        A,
        kernel_size,
        stride,
        pad_size,
        dilation,
        ceil_mode,
        expected_indices);

    auto result_gradinp = torch::max_pool2d_with_indices_backward(
        hB,
        hA,
        kernel_size,
        stride,
        pad_size,
        dilation,
        ceil_mode,
        result_indices);
    EXPECT_EQ(
        allclose(
            result_gradinp.to(torch::kCPU), expected_gradinp, 0.001, 0.001),
        true);
  }
  // Dynamic Maxpool Fwd +BWD
  if (1) {
    torch::Tensor A = torch::randn(
        {2, 16, 41, 123}, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor B = torch::randn(
        {2, 16, 21, 62}, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    std::vector<int64_t> kernel_size = {{3, 3}};
    std::vector<int64_t> stride = {{2, 2}};
    std::vector<int64_t> pad_size = {{1, 1}};
    std::vector<int64_t> dilation = {{1, 1}};
    bool ceil_mode = false;

    auto maxpool_cpu = torch::max_pool2d_with_indices(
        A, kernel_size, stride, pad_size, dilation, ceil_mode);
    auto maxpool_hpu = torch::max_pool2d_with_indices(
        hA, kernel_size, stride, pad_size, dilation, ceil_mode);

    auto expected_indices = std::get<1>(maxpool_cpu);
    auto result_indices = std::get<1>(maxpool_hpu);
    auto max_pool_fwd_out_0 = std::get<0>(maxpool_cpu);

    auto expected_gradinp = torch::max_pool2d_with_indices_backward(
        B,
        A,
        kernel_size,
        stride,
        pad_size,
        dilation,
        ceil_mode,
        expected_indices);

    auto result_gradinp = torch::max_pool2d_with_indices_backward(
        hB,
        hA,
        kernel_size,
        stride,
        pad_size,
        dilation,
        ceil_mode,
        result_indices);
    EXPECT_EQ(
        allclose(
            result_gradinp.to(torch::kCPU), expected_gradinp, 0.001, 0.001),
        true);
  }
}

TEST_F(LazyDynamicFallbackTest, DynamicConvBkwdTest) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
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
    torch::Tensor out_conv = torch::conv2d(
        in_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    auto cpu_out = torch::relu(out_conv);

    // fwd propgation
    torch::Tensor h_weight_tensor = weight_tensor.to(torch::kHPU);
    torch::Tensor h_weight_tensor_hwck = h_weight_tensor;

    torch::Tensor h_in_tensor = in_tensor.to(torch::kHPU);
    torch::Tensor h_out_conv = torch::conv2d(
        h_in_tensor, h_weight_tensor_hwck, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor hpu_out = torch::relu(h_out_conv);

    // bwd propgation with dummy grad tensor
    auto grad_tensor =
        torch::randn({N, C, H - 2, W - 2}, torch::requires_grad(true));
    torch::Tensor tHabanaG = grad_tensor.to(torch::kHPU);
    hpu_out.backward({tHabanaG}, false, true);

    auto out_cpu_lazy = hpu_out.to(torch::kCPU);
    ASSERT_TRUE(torch::allclose(out_cpu_lazy, cpu_out));
  }
}

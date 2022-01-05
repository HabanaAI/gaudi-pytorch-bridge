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

// In this class the pass fallback is enabld and compilation fallback is
// disabled
class LazyDynamicFallbackTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    SetLazyMode();

    SetSeed();

    DisableCpuFallback();

    SetDynamicMode();

    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
  }

  void TearDown() override {
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
    UnsetDynamicMode();

    RestoreMode();
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
    if (!habana_lazy::exec::OptPassCfg::GetInstance()
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

TEST_F(LazyDynamicFallbackTest, MaskRcnnAsStridedTest) {
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
}

// Its places here only because its getting bucket hit and max calculation fail.
TEST_F(LazyDynamicFallbackTest, ViewTest) {
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
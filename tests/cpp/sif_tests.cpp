#include <gtest/gtest.h>
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
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;
using namespace at;

class SifTest : public habana_lazy_test::LazyTest {
 protected:
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

  void validate_shape_start() {
    if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
      SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }

  void validate_shape_end() {
    UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
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
    if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) &&
        !habana_lazy::exec::OptPassCfg::GetInstance()
             ->IsEnabledWeightPermutePass()) {
      h_weight_tensor_hwck = h_weight_tensor.permute({2, 3, 1, 0}).contiguous();
    }

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

// Keeping the following unit tests disabled.
// TODO: Enable the unit tests
TEST_F(SifTest, DISABLED_SingleOpCat) {
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

TEST_F(SifTest, DISABLED_AddMulRelu) {
  validate_shape_start();
  int A = 50;
  const int C = 30;

  int B = 34;
  PT_TEST_DEBUG("PTI_DBG :: TEST ", "  START");

  torch::Tensor h0 =
      torch::randn({C, B, A}, torch::requires_grad(false)).to(torch::kHPU);
  torch::Tensor h1 =
      torch::randn({C, B, A}, torch::requires_grad(false)).to(torch::kHPU);

  torch::Tensor h4 = torch::add(h0, h1);
  torch::Tensor h5 = torch::mul(h0, h1);
  torch::Tensor h6 = torch::mul(h4, h5);
  torch::Tensor h7 = torch::relu(h6);
  auto h7_c = h7.to(torch::kCPU);
  PRINT_TENSOR(h7_c);
  // HbLazyTensor::StepMarker({});

  PT_TEST_DEBUG("PTI_DBG :: TEST ", "  END");
  validate_shape_end();
}

TEST_F(SifTest, DISABLED_ConvTranspose2dBwd) {
  validate_shape_start();

  auto in = torch::randn({64, 4, 28, 28}, torch::requires_grad()); // nchw
  auto hin = in.to(torch::kHPU);
  auto wt = torch::randn({4, 5, 3, 3}, torch::requires_grad()); // ckhw
  auto hwt = wt.to(torch::kHPU);
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) &&
      !habana_lazy::exec::OptPassCfg::GetInstance()
           ->IsEnabledWeightPermutePass()) {
    auto wt_hwck = wt.detach().permute({2, 3, 1, 0}).contiguous();
    hwt = wt_hwck.to(torch::kHPU);
  }
  auto bias = torch::randn({5}, torch::requires_grad()); // k
  auto exp = torch::conv_transpose2d(in, wt, {}, 1, 0, 0, 1, 1);

  auto grad_out = torch::ones_like(exp.detach());
  auto hgrad_out = grad_out.detach().to(torch::kHPU);
  exp.backward(grad_out);
  auto grad_in = in.grad();
  auto grad_wt = wt.grad();

  Tensor hgrad_in, hgrad_wt, hgrad_bias;
  std::array<bool, 3> mask{1, 1, 0};
  std::tie(hgrad_in, hgrad_wt, hgrad_bias) = convolution_backward_hpu_lazy(
      hgrad_out, hin, hwt, {1, 1}, {0, 0}, {1, 1}, true, {0, 0}, 1, mask);

  // TBD: aten::backward is not handled by lazy mode, therefore this is
  // not working. This code can be restored when that is fixed.
  /*auto result = torch::conv_transpose2d(hin, hwt, {}, 1, 0, 0, 1, 1);
  result.backward(hgrad_out);
  auto hgrad_in = hin.grad();
  auto hgrad_wt = hwt.grad();*/

  // without explicit stepmarker here. DMA for hgrad_in tensor gets messed up
  // most likely due to 2 outputs from backward op. TBD: remove this once issue
  // is debugged and fixed.
  HbLazyTensor::StepMarker({});

  auto hgrad_wt_cpu = hgrad_wt.to(torch::kCPU);
  auto hgrad_in_cpu = hgrad_in.to(torch::kCPU);
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) &&
      !habana_lazy::exec::OptPassCfg::GetInstance()
           ->IsEnabledWeightPermutePass()) {
    EXPECT_EQ(
        allclose(grad_wt, hgrad_wt_cpu.permute({3, 2, 0, 1}), 0.01, 0.01),
        true);
  } else {
    EXPECT_EQ(allclose(grad_wt, hgrad_wt_cpu, 0.01, 0.01), true);
  }
  validate_shape_end();
}

// Graph : Reshape + Cat + Relu + Conv2DTransposeBias
// 1. Reshape Op, Has Input Shape Tensor added at Frontend
// 2. Cat Op (can be disabled to try Hybrid mode)
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
TEST_F(SifTest, DISABLED_Reshape_Cat_Relu_Conv2DTransposeBias_Test) {
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
    // 1. Reshape Node
    auto tensor =
        torch::randn({N * C * H * (W / 2)}, torch::requires_grad(false));
    auto reshape_tensor = tensor.reshape({N, C, H, (W / 2)});

    auto h_tensor = tensor.to(torch::kHPU);
    auto h_reshape_tensor = h_tensor.reshape({N, C, H, (W / 2)});

    // 2. Cat Node
    auto tensor_2 =
        torch::randn({N, C, H, (W / 2)}, torch::requires_grad(false));
    auto cat_tensor = torch::cat({reshape_tensor, tensor_2}, 3);

    auto h_tensor_2 = tensor_2.to(torch::kHPU);
    auto h_cat_tensor = torch::cat({h_reshape_tensor, h_tensor_2}, 3);

    // 3. Relu Node
    auto relu_tensor = torch::relu(cat_tensor);
    auto h_relu_tensor = torch::relu(h_cat_tensor);

    // 4. ConvTranpsoseBias Node => Compound Op (Conv + Reshape + Add)
    torch::Tensor bias = torch::randn({C}, torch::dtype(torch::kFloat));
    torch::Tensor h_bias = bias.to(torch::kHPU);
    torch::Tensor weight_tensor =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    torch::Tensor h_weight_tensor = weight_tensor.to(torch::kHPU);
    torch::Tensor h_weight_tensor_hwck = h_weight_tensor;
    if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) &&
        !habana_lazy::exec::OptPassCfg::GetInstance()
             ->IsEnabledWeightPermutePass()) {
      h_weight_tensor_hwck = h_weight_tensor.permute({2, 3, 1, 0}).contiguous();
    }
    torch::Tensor h_out_conv = torch::conv_transpose2d(
        h_relu_tensor, h_weight_tensor_hwck, h_bias, 1, 0, 0, 1, 1);
    torch::Tensor out_conv = torch::conv_transpose2d(
        relu_tensor, weight_tensor, bias, 1, 0, 0, 1, 1);

    torch::Tensor out_conv_hpu = h_out_conv.to(torch::kCPU);
    EXPECT_EQ(allclose(out_conv_hpu, out_conv, 0.01, 0.01), true);
    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
  validate_shape_end();
}

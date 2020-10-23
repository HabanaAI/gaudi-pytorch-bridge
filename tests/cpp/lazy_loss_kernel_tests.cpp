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

using namespace habana_lazy;

class LazyLossKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyLossKernelTest, MseLossTest) {
  torch::Tensor input = torch::randn({3, 5});
  torch::Tensor target = torch::randn({3, 5});
  torch::Tensor grad_input = torch::randn({3, 5});

  auto hinput = input.to(torch::kHABANA);
  auto htarget = target.to(torch::kHABANA);
  auto hgrad_input = grad_input.to(torch::kHABANA);
  torch::Tensor hout1 = torch::mse_loss(hinput, htarget, at::Reduction::None);
  torch::Tensor hout2 = torch::mse_loss_backward(
      hgrad_input, hinput, htarget, at::Reduction::None);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hout1),
                                       GetHbLazyTensor(hout2)};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

  auto out1 = hout1.to(torch::kCPU);
  auto out2 = hout2.to(torch::kCPU);

  auto exp1 = mse_loss(input, target, at::Reduction::None);
  auto exp2 = mse_loss_backward(grad_input, input, target, at::Reduction::None);

  EXPECT_EQ(allclose(out1, exp1), true);
  EXPECT_EQ(allclose(out2, exp2), true);
}

TEST_F(LazyLossKernelTest, NllLossFwdTest) {
  torch::Tensor input = torch::randn({10, 4}, torch::requires_grad(true));
  torch::Tensor hinput = input.to(torch::kHABANA);

  auto target = torch::randint(
      0,
      3,
      {
          10,
      },
      torch::kLong);
  torch::Tensor htarget = target.to(torch::kHABANA);

  torch::nn::NLLLoss loss;
  auto output_cpu = loss->forward(input, target);
  auto output = loss->forward(hinput, htarget);

  Tensor output_hpu = output.to(torch::kCPU);
  EXPECT_EQ(allclose(output_cpu, output_hpu), true);
}

TEST_F(LazyLossKernelTest, NllLossBwdTest) {
  torch::Tensor input = torch::randn({10, 4}, torch::requires_grad(true));
  torch::Tensor hinput = input.to(torch::kHABANA);

  auto target = torch::randint(
      0,
      3,
      {
          10,
      },
      torch::kLong);
  torch::Tensor htarget = target.to(torch::kHABANA);

  auto grad_out = torch::tensor({1}, torch::kFloat);
  torch::Tensor hgrad_out = grad_out.to(torch::kHABANA);

  // HPU kernel does not use this tensor, but we need to create it because
  // "nll_loss_backward" does not compile without this argument. Note that dim &
  // values in this tensor may need to be changed for other "reduction" modes.
  auto sum_weights = torch::tensor({10}, torch::kFloat);
  torch::Tensor hsum_weights = sum_weights.to(torch::kHABANA);

  auto grad_in_cpu = torch::nll_loss_backward(
      grad_out, input, target, {}, 1, -100, sum_weights);
  auto grad_in = torch::nll_loss_backward(
      hgrad_out, hinput, htarget, {}, 1, -100, hsum_weights);

  Tensor grad_in_hpu = grad_in.to(torch::kCPU);
  EXPECT_EQ(allclose(grad_in_cpu, grad_in_hpu), true);
}

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

class LazyNormKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyNormKernelTest, LayerNormForwardExecute) {
  auto input_tensor =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 3, 4, 4}); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  at::Tensor weight =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw;
  torch::Tensor tWeight = weight.to(torch::kHABANA);
  at::Tensor bias =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw;
  torch::Tensor tBias = bias.to(torch::kHABANA);
  auto results =
      torch::native_layer_norm(tHabanaX, tWeight, tBias, 10, 48, 0.01);

  at::Tensor result_lazy = (std::get<0>(results)).to(torch::kCPU);
  auto results_cpu =
      torch::native_layer_norm(input_tensor, weight, bias, 10, 48, 0.01);
  at::Tensor result_cpu = std::get<0>(results_cpu);
  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
}

TEST_F(LazyNormKernelTest, LayerNormBackwardExecute) {
  auto input_grad =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 3, 4, 4}); // nchw
  torch::Tensor tHabanaGrad = input_grad.to(torch::kHABANA);
  auto input =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 3, 4, 4}); // nchw
  torch::Tensor tHabanaIn = input.to(torch::kHABANA);
  auto mean =
      torch::arange(10, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 1});
  auto var = torch::arange(10, torch::dtype(torch::kFloat).requires_grad(false))
                 .reshape({10, 1});
  torch::Tensor tHabanaMean = mean.to(torch::kHABANA);
  torch::Tensor tHabanaVar = var.to(torch::kHABANA);
  auto gamma =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw
  torch::Tensor tGamma = gamma.to(torch::kHABANA);
  auto results = torch::native_layer_norm_backward(
      tHabanaGrad,
      tHabanaIn,
      tHabanaMean,
      tHabanaVar,
      tGamma,
      10,
      48,
      {true, true, true});

  at::Tensor result_lazy = (std::get<0>(results)).to(torch::kCPU);

  auto results_cpu = torch::native_layer_norm_backward(
      input_grad, input, mean, var, gamma, 10, 48, {true, true, true});
  at::Tensor result_cpu = std::get<0>(results_cpu);
  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
}

TEST_F(LazyNormKernelTest, NormScalarTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = torch::norm(hA, 1);
  torch::Tensor Out = torch::norm(A, 1);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}
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

TEST_F(LazyNormKernelTest, BatchNormForwardExecute) {
  auto input_tensor = torch::randn(
      {10, 3, 4, 2}, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  at::Tensor weight =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tWeight = weight.to(torch::kHABANA);
  at::Tensor bias =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tBias = bias.to(torch::kHABANA);
  auto mean = torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaMean = mean.to(torch::kHABANA);
  auto var = torch::ones(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaVar = var.to(torch::kHABANA);

  float mom = 0.1;
  float eps = 1e-5;
  // Training = True
  auto results_cpu = torch::native_batch_norm(
      input_tensor, weight, bias, mean, var, true, mom, eps);

  at::Tensor result_cpu = std::get<0>(results_cpu);
  auto curr_mean_cpu = std::get<1>(results_cpu);

  auto results = torch::native_batch_norm(
      tHabanaX, tWeight, tBias, tHabanaMean, tHabanaVar, true, mom, eps);

  HbLazyTensor::StepMarker({});
  at::Tensor result_lazy = std::get<0>(results).to(torch::kCPU);
  auto curr_mean_lazy = std::get<1>(results).to(torch::kCPU);

  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
  EXPECT_EQ(allclose(curr_mean_lazy.cpu(), curr_mean_cpu, 0.01, 0.01), true);
  EXPECT_EQ(allclose(tHabanaMean.cpu(), mean, 0.01, 0.01), true);
  // Note higher tolerance needed for variance due to TPC kernel accuracy
  // limitation
  EXPECT_EQ(allclose(tHabanaVar.cpu(), var, 0.1, 0.1), true);
}

TEST_F(LazyNormKernelTest, BatchNormInferenceExecute) {
  auto input_tensor = torch::randn(
      {5, 3, 7, 2}, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  at::Tensor weight =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tWeight = weight.to(torch::kHABANA);
  at::Tensor bias =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tBias = bias.to(torch::kHABANA);
  auto mean = torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaMean = mean.to(torch::kHABANA);
  auto var = torch::ones(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaVar = var.to(torch::kHABANA);

  float mom = 0.1;
  float eps = 1e-5;

  // Training = False
  auto results_cpu = torch::native_batch_norm(
      input_tensor, weight, bias, mean, var, false, mom, eps);
  auto result_cpu = std::get<0>(results_cpu);

  auto results = torch::native_batch_norm(
      tHabanaX, tWeight, tBias, tHabanaMean, tHabanaVar, false, mom, eps);

  HbLazyTensor::StepMarker({});
  auto result_lazy = std::get<0>(results).to(torch::kCPU);

  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.001, 0.001), true);
}

TEST_F(LazyNormKernelTest, BatchNormBackwardExecute) {
  auto grad_tensor =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .resize_({10, 3, 4, 4}, c10::MemoryFormat::Contiguous); // nchw
  torch::Tensor tHabanaGrad = grad_tensor.to(torch::kHABANA);
  auto input_tensor =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .resize_({10, 3, 4, 4}, c10::MemoryFormat::Contiguous); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  at::Tensor weight =
      torch::arange(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tWeight = weight.to(torch::kHABANA);
  auto mean =
      torch::arange(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaMean = mean.to(torch::kHABANA);
  auto var = torch::arange(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaVar = var.to(torch::kHABANA);

  auto save_mean =
      torch::arange(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaSaveMean = save_mean.to(torch::kHABANA);
  auto save_ivar =
      torch::arange(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaSaveIVar = save_ivar.to(torch::kHABANA);

  auto results_cpu = torch::native_batch_norm_backward(
      grad_tensor,
      input_tensor,
      weight,
      mean,
      var,
      save_mean,
      save_ivar,
      true,
      0.1,
      {true, true, true});
  at::Tensor result_cpu = std::get<0>(results_cpu);

  auto results = torch::native_batch_norm_backward(
      tHabanaGrad,
      tHabanaX,
      tWeight,
      tHabanaMean,
      tHabanaVar,
      tHabanaSaveMean,
      tHabanaSaveIVar,
      true,
      0.1,
      {true, true, true});

  at::Tensor result_lazy = std::get<0>(results).to(torch::kCPU);
  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
}

TEST_F(LazyNormKernelTest, NormScalarTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = torch::norm(hA, 1);
  torch::Tensor Out = torch::norm(A, 1);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}
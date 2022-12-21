#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
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
using namespace at;

class LazyNormKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyNormKernelTest, LayerNormForwardExecute) {
  auto input_tensor =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 1, 3, 4, 4}); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  at::Tensor weight =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw;
  torch::Tensor tWeight = weight.to(torch::kHPU);
  at::Tensor bias =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw;
  torch::Tensor tBias = bias.to(torch::kHPU);
  auto results =
      torch::native_layer_norm(tHabanaX, {1, 3, 4, 4}, tWeight, tBias, 0.01);

  at::Tensor result_lazy = (std::get<0>(results)).to(torch::kCPU);
  auto results_cpu =
      torch::native_layer_norm(input_tensor, {1, 3, 4, 4}, weight, bias, 0.01);
  at::Tensor result_cpu = std::get<0>(results_cpu);
  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
}

TEST_F(LazyNormKernelTest, InstanceNormChLast) {
  auto input_tensor =
      torch::arange(240, torch::dtype(torch::kFloat).requires_grad(false))
          .resize_({10, 3, 4, 2}, c10::MemoryFormat::ChannelsLast);
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);

  at::Tensor weight =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tWeight = weight.to(torch::kHPU);

  at::Tensor bias =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tBias = bias.to(torch::kHPU);

  auto mean = torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaMean = mean.to(torch::kHPU);

  auto var = torch::ones(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaVar = var.to(torch::kHPU);

  constexpr float mom = 0.1;
  constexpr float eps = 1e-5;
  auto result_cpu = torch::instance_norm(
      input_tensor, weight, bias, mean, var, true, mom, eps, false);

  auto result_lazy = torch::instance_norm(
      tHabanaX, tWeight, tBias, tHabanaMean, tHabanaVar, true, mom, eps, false);

  EXPECT_EQ(allclose(result_lazy.to("cpu"), result_cpu, 0.01, 0.01), true);
}

TEST_F(LazyNormKernelTest, DISABLED_InstanceNorm3dFwdBwd) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
    GTEST_SKIP();
  }
  auto batch_dim = 1;
  auto channel_dim = 320;
  auto depth_dim = 8;
  auto height_dim = 8;
  auto width_dim = 8;
  auto num_el = batch_dim * channel_dim * depth_dim * height_dim * width_dim;
  auto input_tensor = torch::randn(
      {batch_dim, channel_dim, depth_dim, height_dim, width_dim},
      torch::dtype(torch::kFloat).requires_grad(true));
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU).detach();
  tHabanaX.requires_grad_(true);

  at::Tensor weight = torch::randn(
      channel_dim, torch::dtype(torch::kFloat).requires_grad(true));
  torch::Tensor tWeight = weight.to(torch::kHPU).detach();
  tWeight.requires_grad_(true);

  at::Tensor bias = torch::randn(
      channel_dim, torch::dtype(torch::kFloat).requires_grad(true));
  torch::Tensor tBias = bias.to(torch::kHPU).detach();
  tBias.requires_grad_(true);

  auto mean = torch::randn(
      channel_dim, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaMean = mean.to(torch::kHPU);

  auto var = torch::ones(
      channel_dim, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaVar = var.to(torch::kHPU);

  constexpr float mom = 0.1;
  constexpr float eps = 1e-5;
  auto result_cpu = torch::instance_norm(
      input_tensor, weight, bias, mean, var, true, mom, eps, false);
  auto grad_out = torch::ones_like(result_cpu);
  auto hgrad_out = grad_out.to(torch::kHPU).detach();
  hgrad_out.requires_grad_(true);
  result_cpu.backward(grad_out);
  auto grad_in = input_tensor.grad();

  auto result_lazy = torch::instance_norm(
      tHabanaX, tWeight, tBias, tHabanaMean, tHabanaVar, true, mom, eps, false);
  auto result_lazy_hpu = result_lazy.to(torch::kCPU);
  result_lazy.backward(hgrad_out);
  auto hgrad_in = tHabanaX.grad();
  auto hgrad_in_hpu = hgrad_in.to(torch::kCPU);

  EXPECT_EQ(
      allclose(result_lazy_hpu, result_cpu, 0.01, 0.01) &&
          allclose(hgrad_in_hpu, grad_in, 0.01, 0.01),
      true);
}

TEST_F(LazyNormKernelTest, InstanceNorm3dChLastFwdBwd) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
    GTEST_SKIP();
  }
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  auto batch_dim = 1;
  auto channel_dim = 320;
  auto depth_dim = 8;
  auto height_dim = 8;
  auto width_dim = 8;
  auto num_el = batch_dim * channel_dim * depth_dim * height_dim * width_dim;
  auto input_tensor = torch::randn(
      {batch_dim, channel_dim, depth_dim, height_dim, width_dim},
      torch::dtype(torch::kFloat).requires_grad(true));
  torch::Tensor tHabanaX =
      input_tensor.contiguous(c10::MemoryFormat::ChannelsLast3d)
          .to(torch::kHPU)
          .detach();
  tHabanaX.requires_grad_(true);

  at::Tensor weight = torch::randn(
      channel_dim, torch::dtype(torch::kFloat).requires_grad(true));
  torch::Tensor tWeight = weight.to(torch::kHPU).detach();
  tWeight.requires_grad_(true);

  at::Tensor bias = torch::randn(
      channel_dim, torch::dtype(torch::kFloat).requires_grad(true));
  torch::Tensor tBias = bias.to(torch::kHPU).detach();
  tBias.requires_grad_(true);

  auto mean = torch::randn(
      channel_dim, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaMean = mean.to(torch::kHPU);

  auto var = torch::ones(
      channel_dim, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaVar = var.to(torch::kHPU);

  constexpr float mom = 0.1;
  constexpr float eps = 1e-5;
  auto result_cpu = torch::instance_norm(
      input_tensor, weight, bias, mean, var, true, mom, eps, false);
  auto grad_out = torch::ones_like(result_cpu);
  auto hgrad_out = grad_out.to(torch::kHPU).detach();
  hgrad_out.requires_grad_(true);
  result_cpu.backward(grad_out);
  auto grad_in = input_tensor.grad();

  auto result_lazy = torch::instance_norm(
      tHabanaX, tWeight, tBias, tHabanaMean, tHabanaVar, true, mom, eps, false);
  auto result_lazy_hpu = result_lazy.to(torch::kCPU);
  result_lazy.backward(hgrad_out);
  auto hgrad_in = tHabanaX.grad();
  auto hgrad_in_hpu = hgrad_in.to(torch::kCPU);

  EXPECT_EQ(
      allclose(result_lazy_hpu, result_cpu, 0.01, 0.01) &&
          allclose(hgrad_in_hpu, grad_in, 0.01, 0.01),
      true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyNormKernelTest, LayerNormBackwardExecute) {
  auto input_grad =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 1, 3, 4, 4}); // nchw
  torch::Tensor tHabanaGrad = input_grad.to(torch::kHPU);
  auto input =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 1, 3, 4, 4}); // nchw
  torch::Tensor tHabanaIn = input.to(torch::kHPU);
  auto mean =
      torch::arange(10, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 1});
  auto var = torch::arange(10, torch::dtype(torch::kFloat).requires_grad(false))
                 .reshape({10, 1});
  torch::Tensor tHabanaMean = mean.to(torch::kHPU);
  torch::Tensor tHabanaVar = var.to(torch::kHPU);
  auto gamma =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw
  torch::Tensor tGamma = gamma.to(torch::kHPU);
  auto bias =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw
  torch::Tensor tBias = bias.to(torch::kHPU);

  auto results = torch::native_layer_norm_backward(
      tHabanaGrad,
      tHabanaIn,
      {1, 3, 4, 4},
      tHabanaMean,
      tHabanaVar,
      tGamma,
      tBias,
      {true, true, true});

  auto results_cpu = torch::native_layer_norm_backward(
      input_grad,
      input,
      {1, 3, 4, 4},
      mean,
      var,
      gamma,
      bias,
      {true, true, true});
  at::Tensor result_lazy = (std::get<0>(results)).to(torch::kCPU);

  at::Tensor result_cpu = std::get<0>(results_cpu);
  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
}

TEST_F(LazyNormKernelTest, LayerNormBackwardExecute_gal) {
  auto input_grad =
      torch::ones({24}, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({2, 3, 4}); // nchw
  torch::Tensor tHabanaGrad = input_grad.to(torch::kHPU);
  auto input =
      torch::arange(24, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({2, 3, 4}); // nchw
  torch::Tensor tHabanaIn = input.to(torch::kHPU);
  auto mean = torch::ones({6}, torch::dtype(torch::kFloat).requires_grad(false))
                  .reshape({2, 3});
  auto var = torch::ones({6}, torch::dtype(torch::kFloat).requires_grad(false))
                 .reshape({2, 3});
  torch::Tensor tHabanaMean = mean.to(torch::kHPU);
  torch::Tensor tHabanaVar = var.to(torch::kHPU);

  auto gamma =
      torch::ones({4}, torch::dtype(torch::kFloat).requires_grad(false));
  // .reshape({4});
  torch::Tensor tGamma = gamma.to(torch::kHPU);
  auto bias =
      torch::ones({4}, torch::dtype(torch::kFloat).requires_grad(false));
  // .reshape({4});
  torch::Tensor tBias = bias.to(torch::kHPU);

  auto results_cpu = torch::native_layer_norm_backward(
      input_grad, input, 4, mean, var, gamma, bias, {true, true, true});

  auto results = torch::native_layer_norm_backward(
      tHabanaGrad,
      tHabanaIn,
      4,
      tHabanaMean,
      tHabanaVar,
      tGamma,
      tBias,
      {true, true, true});

  at::Tensor result_lazy = (std::get<0>(results)).to(torch::kCPU);

  at::Tensor result_cpu = std::get<0>(results_cpu);

  // Print Both Tensor contents:
  //   std::clog << "\n\n\n\nLazy Result: " << std::endl;
  //   PrintATenTensor(result_lazy);
  //   std::clog << result_lazy << std::endl;
  //   std::clog << "\n\n\n\nCPU Result: " << std::endl;
  //   PrintATenTensor(result_cpu);
  //   std::clog << result_cpu << std::endl;

  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
}

// Also validates ComputeOutputShape for BatchNormFwd
TEST_F(LazyNormKernelTest, BatchNormForwardExecute) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  auto input_tensor = torch::randn(
      {10, 3, 4, 2}, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  at::Tensor weight =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tWeight = weight.to(torch::kHPU);
  at::Tensor bias =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tBias = bias.to(torch::kHPU);
  auto mean = torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaMean = mean.to(torch::kHPU);
  auto var = torch::ones(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaVar = var.to(torch::kHPU);

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
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyNormKernelTest, BatchNorm7DForwardExecute) {
  auto input_tensor = torch::randn(
      {1, 5, 8, 10, 3, 4, 2}, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  at::Tensor weight =
      torch::randn(5, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tWeight = weight.to(torch::kHPU);
  at::Tensor bias =
      torch::randn(5, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tBias = bias.to(torch::kHPU);
  auto mean = torch::randn(5, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaMean = mean.to(torch::kHPU);
  auto var = torch::ones(5, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaVar = var.to(torch::kHPU);

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

TEST_F(LazyNormKernelTest, BatchNorm5DChannelsLastForwardExecute) {
  auto input_tensor =
      torch::randn(
          {8, 3, 10, 10, 4}, torch::dtype(torch::kFloat).requires_grad(false))
          .contiguous(c10::MemoryFormat::ChannelsLast3d);
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  at::Tensor weight =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tWeight = weight.to(torch::kHPU);
  at::Tensor bias =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tBias = bias.to(torch::kHPU);
  auto mean = torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaMean = mean.to(torch::kHPU);
  auto var = torch::ones(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaVar = var.to(torch::kHPU);

  float mom = 0.1;
  float eps = 1e-5;
  // Training = True
  auto results_cpu = torch::native_batch_norm(
      input_tensor, weight, bias, mean, var, true, mom, eps);

  at::Tensor result_cpu = std::get<0>(results_cpu);
  auto curr_mean_cpu = std::get<1>(results_cpu);
  auto curr_var_cpu = std::get<2>(results_cpu);

  auto results = torch::native_batch_norm(
      tHabanaX, tWeight, tBias, tHabanaMean, tHabanaVar, true, mom, eps);

  HbLazyTensor::StepMarker({});
  at::Tensor result_lazy = std::get<0>(results).to(torch::kCPU);
  auto curr_mean_lazy = std::get<1>(results).to(torch::kCPU);
  auto curr_var_lazy = std::get<2>(results).to(torch::kCPU);

  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
  EXPECT_EQ(allclose(curr_mean_lazy.cpu(), curr_mean_cpu, 0.01, 0.01), true);
  EXPECT_EQ(allclose(curr_var_lazy.cpu(), curr_var_cpu, 0.01, 0.01), true);
}

TEST_F(LazyNormKernelTest, BatchNormAffineFalseForwardExecute) {
  auto input_tensor = torch::randn(
      {10, 3, 4, 2}, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  auto mean = torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaMean = mean.to(torch::kHPU);
  auto var = torch::ones(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaVar = var.to(torch::kHPU);

  float mom = 0.1;
  float eps = 1e-5;
  // Training = True
  auto results_cpu = torch::native_batch_norm(
      input_tensor, c10::nullopt, c10::nullopt, mean, var, true, mom, eps);
  at::Tensor result_cpu = std::get<0>(results_cpu);
  auto curr_mean_cpu = std::get<1>(results_cpu);
  auto results = torch::native_batch_norm(
      tHabanaX,
      c10::nullopt,
      c10::nullopt,
      tHabanaMean,
      tHabanaVar,
      true,
      mom,
      eps);

  at::Tensor result_lazy = std::get<0>(results).to(torch::kCPU);
  auto curr_mean_lazy = std::get<1>(results).to(torch::kCPU);
  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
  EXPECT_EQ(allclose(curr_mean_lazy.cpu(), curr_mean_cpu, 0.01, 0.01), true);
  EXPECT_EQ(allclose(tHabanaMean.cpu(), mean, 0.01, 0.01), true);
  // Note higher tolerance needed for variance due to TPC kernel accuracy
  // limitation
  EXPECT_EQ(allclose(tHabanaVar.cpu(), var, 0.1, 0.1), true);
}

// Also validates ComputeOutputShape for BatchNormInf
TEST_F(LazyNormKernelTest, BatchNormInferenceExecute) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  auto input_tensor = torch::randn(
      {5, 3, 7, 2}, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  at::Tensor weight =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tWeight = weight.to(torch::kHPU);
  at::Tensor bias =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tBias = bias.to(torch::kHPU);
  auto mean = torch::zeros(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaMean = mean.to(torch::kHPU);
  auto var = torch::ones(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaVar = var.to(torch::kHPU);

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
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates ComputeOutputShape for BatchNormBwd
TEST_F(LazyNormKernelTest, BatchNormBackwardExecute) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  auto grad_tensor =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .resize_({10, 3, 4, 4}, c10::MemoryFormat::Contiguous); // nchw
  torch::Tensor tHabanaGrad = grad_tensor.to(torch::kHPU);
  auto input_tensor =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .resize_({10, 3, 4, 4}, c10::MemoryFormat::Contiguous); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  at::Tensor weight =
      torch::arange(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tWeight = weight.to(torch::kHPU);
  auto mean =
      torch::arange(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaMean = mean.to(torch::kHPU);
  auto var = torch::arange(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaVar = var.to(torch::kHPU);

  auto save_mean =
      torch::arange(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaSaveMean = save_mean.to(torch::kHPU);
  auto save_ivar =
      torch::arange(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaSaveIVar = save_ivar.to(torch::kHPU);

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
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyNormKernelTest, InstanceNorm) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  auto input_tensor = torch::randn(
      {10, 3, 4, 2}, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  at::Tensor weight =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tWeight = weight.to(torch::kHPU);
  at::Tensor bias =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tBias = bias.to(torch::kHPU);
  auto mean = torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaMean = mean.to(torch::kHPU);
  auto var = torch::ones(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaVar = var.to(torch::kHPU);

  constexpr float mom = 0.1;
  constexpr float eps = 1e-5;
  auto result_cpu = torch::instance_norm(
      input_tensor, weight, bias, mean, var, true, mom, eps, false);

  auto result_lazy = torch::instance_norm(
      tHabanaX, tWeight, tBias, tHabanaMean, tHabanaVar, true, mom, eps, false);

  HbLazyTensor::StepMarker({});

  EXPECT_EQ(allclose(result_lazy.to("cpu"), result_cpu, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyNormKernelTest, InstanceNormNormv) {
  auto input_tensor = torch::randn(
      {10, 3, 4, 2}, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  at::Tensor weight =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tWeight = weight.to(torch::kHPU);
  at::Tensor bias =
      torch::randn(3, torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tBias = bias.to(torch::kHPU);

  constexpr float mom = 0.1;
  constexpr float eps = 1e-5;
  auto result_cpu = torch::instance_norm(
      input_tensor,
      weight,
      bias,
      torch::Tensor(),
      torch::Tensor(),
      true,
      mom,
      eps,
      false);

  auto result_lazy = torch::instance_norm(
      tHabanaX,
      tWeight,
      tBias,
      torch::Tensor(),
      torch::Tensor(),
      true,
      mom,
      eps,
      false);

  EXPECT_EQ(allclose(result_lazy.to("cpu"), result_cpu, 0.01, 0.01), true);
}

TEST_F(LazyNormKernelTest, NormScalarTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::norm(hA, 1);
  torch::Tensor Out = torch::norm(A, 1);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, FusedNormTest) {
  std::vector<torch::Tensor> grad_vec;
  std::vector<torch::Tensor> grad_vec_h;
  std::vector<torch::Tensor> grad_vec_norms;
  auto num_params = 4;
  // setup input grad tensor lists
  for (auto i = 0; i < num_params; i++) {
    auto t = torch::randn({2, 2});
    grad_vec.push_back(t);
    grad_vec_norms.push_back(torch::norm(t));
    auto tH = t.to(torch::kHPU);
    grad_vec_h.push_back(tH);
  }
  // init max_norm
  torch::Tensor max_norm =
      torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat32)) * 1.0;
  auto max_norm_hpu = max_norm.to(torch::kHPU);
  // do hpu and cpu fused_norm calcs
  auto total_norm_hpu = fused_norm_hpu_wrap(grad_vec_h, max_norm_hpu, 2.0);
  auto total_norm_cpu = torch::norm(torch::stack(grad_vec_norms));

  EXPECT_EQ(
      allclose(total_norm_hpu.to(torch::kCPU), total_norm_cpu, 0.0001), true);
}

TEST_F(LazyNormKernelTest, FusedNormViewTest) {
  std::vector<torch::Tensor> grad_vec;
  std::vector<torch::Tensor> grad_vec_h;
  std::vector<torch::Tensor> grad_vec_norms;

  // setup input grad tensor lists
  auto t = torch::randn({2, 2});
  auto tH = t.to(torch::kHPU);
  auto v = t.view(-1);
  grad_vec.push_back(v);
  grad_vec_norms.push_back(torch::norm(v));

  auto hV = tH.view(-1);
  grad_vec_h.push_back(hV);

  // init max_norm
  torch::Tensor max_norm =
      torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat32)) * 1.0;
  auto max_norm_hpu = max_norm.to(torch::kHPU);
  // do hpu and cpu fused_norm calcs
  auto total_norm_hpu = fused_norm_hpu_wrap(grad_vec_h, max_norm_hpu, 2.0);
  auto total_norm_cpu = torch::norm(torch::stack(grad_vec_norms));

  HbLazyTensor::StepMarker({});

  EXPECT_EQ(
      allclose(total_norm_hpu.to(torch::kCPU), total_norm_cpu, 0.001), true);
}

TEST_F(LazyNormKernelTest, BatchNormBackwardAdd) {
  auto grad_tensor = torch::randn({10, 3, 4, 4}, torch::requires_grad(false));
  auto tHabanaGrad = grad_tensor.to(torch::kHPU);

  auto input_tensor = torch::randn({10, 3, 4, 4}, torch::requires_grad(false));
  auto tHabanaX = input_tensor.to(torch::kHPU);

  auto weight = torch::randn({3}, torch::requires_grad(false));
  auto tWeight = weight.to(torch::kHPU);

  auto mean = torch::randn({3}, torch::requires_grad(false));
  auto tHabanaMean = mean.to(torch::kHPU);

  auto var = torch::randn({3}, torch::requires_grad(false));
  auto tHabanaVar = var.to(torch::kHPU);

  auto save_mean = torch::randn({3}, torch::requires_grad(false));
  auto tHabanaSaveMean = save_mean.to(torch::kHPU);

  auto save_ivar = torch::randn({3}, torch::requires_grad(false));
  auto tHabanaSaveIVar = save_ivar.to(torch::kHPU);

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

  auto cpu_results = torch::native_batch_norm_backward(
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
  tWeight = tWeight.add_(std::get<1>(results));
  weight = weight.add_(std::get<1>(cpu_results));

  EXPECT_EQ(allclose(tWeight.to(torch::kCPU), weight, 0.0001), true);
}

TEST_F(LazyNormKernelTest, FrobNormDimTest) {
  torch::Tensor A = torch::randn({2, 3, 4, 5}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::frobenius_norm(hA, {1, 3}, false);
  torch::Tensor Out = torch::frobenius_norm(A, {1, 3}, false);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, L0NormScalarTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::norm(hA, 0);
  torch::Tensor Out = torch::norm(A, 0);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, NormScalarDimTest) {
  torch::Tensor A = torch::randn({2, 2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  std::vector<int64_t> dimarr{2, 1, 0};
  c10::IntArrayRef dims(dimarr.data(), dimarr.size());
  torch::Tensor hOut = torch::norm(hA, 1, dimarr, false);
  torch::Tensor Out = torch::norm(A, 1, dimarr, false);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, NormScalarDimDtypeTest) {
  torch::Tensor A = torch::randn({2, 2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  std::vector<int64_t> dimarr{2, 1, 0};
  c10::IntArrayRef dims(dimarr.data(), dimarr.size());
  torch::Tensor hOut = torch::norm(hA, 1, dimarr, false, at::kFloat);
  torch::Tensor Out = torch::norm(A, 1, dimarr, false, at::kFloat);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, NormScalarDimEmptyCasepGenTest) {
  torch::Tensor A = torch::randn({2, 2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  std::vector<int64_t> dimarr{};
  c10::IntArrayRef dims(dimarr.data(), dimarr.size());
  torch::Tensor hOut = torch::norm(hA, 3.0, dimarr, false);
  torch::Tensor Out = torch::norm(A, 3.0, dimarr, false);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, NormScalarDimEmptyCasep0Test) {
  torch::Tensor A = torch::randn({2, 2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  std::vector<int64_t> dimarr{};
  c10::IntArrayRef dims(dimarr.data(), dimarr.size());
  torch::Tensor hOut = torch::norm(hA, 0.0, dimarr, false);
  torch::Tensor Out = torch::norm(A, 0.0, dimarr, false);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, NormScalarDimBFDtypeTest) {
  torch::Tensor A =
      torch::randn({2, 2, 2}, at::device(at::kCPU).dtype(at::kBFloat16));
  torch::Tensor hA = A.to(torch::kHPU);
  std::vector<int64_t> dimarr{2, 1, 0};
  c10::IntArrayRef dims(dimarr.data(), dimarr.size());
  torch::Tensor hOut = torch::norm(hA, 1, dimarr, false, at::kFloat);
  torch::Tensor Out = torch::norm(A, 1, dimarr, false, at::kFloat);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, NormScalarDimEmptyCasepInfTest) {
  torch::Tensor A = torch::randn({2, 2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  std::vector<int64_t> dimarr{};
  c10::IntArrayRef dims(dimarr.data(), dimarr.size());
  torch::Tensor hOut =
      torch::norm(hA, std::numeric_limits<float>::infinity(), dimarr, false);
  torch::Tensor Out =
      torch::norm(A, std::numeric_limits<float>::infinity(), dimarr, false);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}
/*
TEST_F(LazyNormKernelTest, NormScalarNonFloatDimTest) {
  auto options = torch::TensorOptions().dtype(torch::kByte);
  torch::Tensor A = torch::randint(0, 10, {2, 2, 2}, options);
  torch::Tensor hA = A.to(torch::kHPU);
  std::vector<int64_t> dimarr{2, 1, 0};
  c10::IntArrayRef dims(dimarr.data(), dimarr.size());
  torch::Tensor hOut = torch::norm(hA, 1, dimarr, true, c10::ScalarType::Float);
  torch::Tensor Out = torch::norm(A, 1, dimarr, true, c10::ScalarType::Float);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}
*/
TEST_F(LazyNormKernelTest, L0NormScalarDimTest) {
  torch::Tensor A = torch::randn({2, 2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  std::vector<int64_t> dimarr{0, 1};
  c10::IntArrayRef dims(dimarr.data(), dimarr.size());
  torch::Tensor hOut = torch::norm(hA, 0, dimarr, true);
  torch::Tensor Out = torch::norm(A, 0, dimarr, true);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, LInfNormScalarDimTest) {
  torch::Tensor A = torch::randn({2, 2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  std::vector<int64_t> dimarr{1, 0, 2};
  c10::IntArrayRef dims(dimarr.data(), dimarr.size());
  torch::Tensor hOut =
      torch::norm(hA, std::numeric_limits<float>::infinity(), dimarr, false);
  torch::Tensor Out =
      torch::norm(A, std::numeric_limits<float>::infinity(), dimarr, false);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, LNegInfNormScalarDimTest) {
  torch::Tensor A = torch::randn({2, 2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  std::vector<int64_t> dimarr{2, 0, 1};
  c10::IntArrayRef dims(dimarr.data(), dimarr.size());
  torch::Tensor hOut =
      torch::norm(hA, -std::numeric_limits<float>::infinity(), dimarr, true);
  torch::Tensor Out =
      torch::norm(A, -std::numeric_limits<float>::infinity(), dimarr, true);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, NormScalarZeroDimTest) {
  torch::Tensor A = torch::tensor(2.67);
  torch::Tensor hA = A.to(torch::kHPU);
  std::vector<int64_t> dimarr{};
  c10::IntArrayRef dims(dimarr.data(), dimarr.size());
  torch::Tensor Out = torch::norm(A, 10.0);
  torch::Tensor hOut = torch::norm(hA, 10.0);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, NormScalarZSTTest) {
  torch::Tensor A = torch::randn({2, 0, 2, 4}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  std::vector<int64_t> dimarr = {};
  c10::IntArrayRef dims(dimarr.data(), dimarr.size());
  torch::Tensor Out = torch::norm(A, -10.0, dims);
  torch::Tensor hOut = torch::norm(hA, -10.0, dims);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, WeightNormTest) {
  at::Tensor v_in =
      at::randn({1024, 64, 128}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor g_in =
      at::randn({1, 1, 128}, at::device(at::kCPU).dtype(at::kFloat));
  int64_t dim(2);
  // CPU Run
  at::Tensor output_ = at::_weight_norm(v_in, g_in, dim);
  // Prepare HPU inputs
  at::Tensor h_v_in = v_in.to(at::device(at::kHPU));
  at::Tensor h_g_in = g_in.to(at::device(at::kHPU));
  // HPU Run
  at::Tensor h_output_ = at::_weight_norm(h_v_in, h_g_in, dim);
  // Compare CPU vs HPU
  at::Tensor h_output__cpu = h_output_.to(at::device(at::kCPU));
  EXPECT_EQ(allclose(h_output__cpu, output_, 0.0001), true);
}

TEST_F(LazyNormKernelTest, NormScalarDimOutTest) {
  torch::Tensor A = torch::randn({2, 2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  std::vector<int64_t> dimarr{1, 0};
  c10::IntArrayRef dims(dimarr.data(), dimarr.size());
  torch::Tensor Out = torch::empty({2});
  torch::Tensor hOut = Out.to(torch::kHPU);
  torch::norm_out(hOut, hA, 1, dimarr, false);
  torch::norm_out(Out, A, 1, dimarr, false);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, NormScalarDimDtypeOutTest) {
  torch::Tensor A = torch::randn({2, 2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  std::vector<int64_t> dimarr{1, 0};
  torch::Tensor Out = torch::empty({});
  torch::Tensor hOut = Out.to(torch::kHPU);
  c10::IntArrayRef dims(dimarr.data(), dimarr.size());
  torch::norm_out(hOut, hA, 1, dimarr, false, at::kFloat);
  torch::norm_out(Out, A, 1, dimarr, false, at::kFloat);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.0001), true);
}

TEST_F(LazyNormKernelTest, LayerNormFwdBwdExecute) {
  auto input_tensor =
      torch::arange(16, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({2, 1, 2, 4}); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  at::Tensor weight =
      torch::arange(8, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 2, 4}); // nchw;
  torch::Tensor tWeight = weight.to(torch::kHPU);
  at::Tensor bias =
      torch::arange(8, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 2, 4}); // nchw;
  torch::Tensor tBias = bias.to(torch::kHPU);
  auto results =
      torch::native_layer_norm(tHabanaX, {1, 2, 4}, tWeight, tBias, 0.01);

  at::Tensor result_lazy = (std::get<0>(results)).to(torch::kCPU);
  auto results_cpu =
      torch::native_layer_norm(input_tensor, {1, 2, 4}, weight, bias, 0.01);
  at::Tensor result_cpu = std::get<0>(results_cpu);
  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);

  // Backward

  auto input_grad =
      torch::arange(16, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({2, 1, 2, 4}); // nchw
  torch::Tensor tHabanaGrad = input_grad.to(torch::kHPU);
  auto mean = std::get<1>(results_cpu);
  auto var = std::get<2>(results_cpu);
  torch::Tensor tHabanaMean = std::get<1>(results);
  torch::Tensor tHabanaVar = std::get<2>(results);
  auto gamma = weight; // nchw
  torch::Tensor tGamma = tWeight;

  auto results_bwd = torch::native_layer_norm_backward(
      tHabanaGrad,
      tHabanaX,
      {1, 2, 4},
      tHabanaMean,
      tHabanaVar,
      tGamma,
      tBias,
      {true, true, true});

  auto results_bwd_cpu = torch::native_layer_norm_backward(
      input_grad,
      input_tensor,
      {1, 2, 4},
      mean,
      var,
      gamma,
      bias,
      {true, true, true});
  at::Tensor result_bwd_lazy = (std::get<0>(results_bwd)).to(torch::kCPU);

  at::Tensor result_bwd_cpu = std::get<0>(results_bwd_cpu);

  EXPECT_EQ(allclose(result_bwd_lazy, result_bwd_cpu, 0.01, 0.01), true);
}

TEST_F(LazyNormKernelTest, GroupNormFwdBwdExecute) {
  int64_t N = 2;
  int64_t C = 4;
  int64_t H = 2;
  int64_t W = 2;
  int64_t G = 2;
  double eps = 0.0001;
  auto input_tensor =
      torch::ones(
          N * C * H * W, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({N, C, H, W}); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  at::Tensor weight =
      torch::ones(C, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({C}); // nchw;
  torch::Tensor tWeight = weight.to(torch::kHPU);
  at::Tensor bias =
      torch::zeros(C, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({C}); // nchw;
  torch::Tensor tBias = bias.to(torch::kHPU);
  auto results =
      torch::native_group_norm(tHabanaX, tWeight, tBias, N, C, H * W, G, eps);

  at::Tensor result_lazy = std::get<0>(results);
  at::Tensor mean_lazy = std::get<1>(results);
  at::Tensor rstd_lazy = std::get<2>(results);
  auto results_cpu =
      torch::native_group_norm(input_tensor, weight, bias, N, C, H * W, G, eps);
  at::Tensor result_cpu = std::get<0>(results_cpu);
  at::Tensor mean_cpu = std::get<1>(results_cpu);
  at::Tensor rstd_cpu = std::get<2>(results_cpu);

  EXPECT_EQ(
      allclose(result_lazy.to(torch::kCPU), result_cpu, 0.05, 0.05), true);
  EXPECT_EQ(allclose(mean_lazy.to(torch::kCPU), mean_cpu, 0.05, 0.05), true);
  EXPECT_EQ(allclose(rstd_lazy.to(torch::kCPU), rstd_cpu, 0.05, 0.05), true);

  // Backward

  auto input_grad =
      torch::ones(
          N * C * H * W, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({N, C, H, W}); // nchw
  torch::Tensor tHabanaGrad = input_grad.to(torch::kHPU);

  auto results_bwd = torch::native_group_norm_backward(
      tHabanaGrad,
      tHabanaX,
      mean_lazy,
      rstd_lazy,
      tWeight,
      N,
      C,
      H * W,
      G,
      {true, true, true});

  auto results_bwd_cpu = torch::native_group_norm_backward(
      input_grad,
      input_tensor,
      mean_cpu,
      rstd_cpu,
      weight,
      N,
      C,
      H * W,
      G,
      {true, true, true});
  at::Tensor result_bwd_lazy = (std::get<0>(results_bwd)).to(torch::kCPU);
  at::Tensor grad_weight_bwd_lazy = (std::get<1>(results_bwd)).to(torch::kCPU);
  at::Tensor grad_bias_bwd_lazy = (std::get<2>(results_bwd)).to(torch::kCPU);

  at::Tensor result_bwd_cpu = std::get<0>(results_bwd_cpu);
  at::Tensor grad_weight_bwd_cpu = std::get<1>(results_bwd_cpu);
  at::Tensor grad_bias_bwd_cpu = std::get<2>(results_bwd_cpu);
  std::cout << "GN bwd: results cpu & hpu:" << std::endl
            << result_bwd_cpu << std::endl
            << result_bwd_lazy << std::endl;

  EXPECT_EQ(allclose(result_bwd_lazy, result_bwd_cpu, 0.05, 0.05), true);
  std::cout << "GN bwd: grad_weight cpu & hpu:" << std::endl
            << grad_weight_bwd_cpu << std::endl
            << grad_weight_bwd_lazy << std::endl;
  EXPECT_EQ(
      allclose(grad_weight_bwd_lazy, grad_weight_bwd_cpu, 0.05, 0.05), true);
  std::cout << "GN bwd: grad_bias cpu & hpu:" << std::endl
            << grad_bias_bwd_cpu << std::endl
            << grad_bias_bwd_lazy << std::endl;
  EXPECT_EQ(allclose(grad_bias_bwd_lazy, grad_bias_bwd_cpu, 0.05, 0.05), true);
}

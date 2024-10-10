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

class LazySoftmaxKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazySoftmaxKernelTest, LogSoftMaxTest) {
  torch::Tensor input = torch::rand({64, 10}, torch::requires_grad(false));
  torch::Tensor hinput = input.to(torch::kHPU);
  constexpr int dim = 0;
  torch::Tensor hout = torch::log_softmax(hinput, dim);

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(hout)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto hout1 = hout.to(torch::kCPU);

  auto cout = torch::log_softmax(input, dim);

  EXPECT_EQ(allclose(hout1, cout), true);
}

TEST_F(LazySoftmaxKernelTest, LogSoftMaxTest4D) {
  torch::Tensor input =
      torch::rand({14, 4, 192, 160}, torch::requires_grad(false));
  torch::Tensor hinput = input.to(torch::kHPU);
  constexpr int dim = 1;
  torch::Tensor hout = torch::log_softmax(hinput, dim);

  auto hout1 = hout.to(torch::kCPU);

  auto cout = torch::log_softmax(input, dim);

  EXPECT_EQ(allclose(hout1, cout), true);
}

TEST_F(LazySoftmaxKernelTest, CrossEntropyTest) {
  torch::Tensor input_tensor =
      torch::rand({16, 32, 12, 10}, torch::requires_grad(false));
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);

  torch::Tensor weight_tensor =
      torch::rand({4, 32, 1, 1}, torch::requires_grad(false));
  torch::Tensor tHabanaW = weight_tensor.to(torch::kHPU);

  auto target = torch::randint(0, 3, {16, 12, 10}, torch::kLong);
  torch::Tensor htarget = target.to(torch::kHPU);

  torch::Tensor houtConv =
      torch::conv2d(tHabanaX, tHabanaW, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::nn::CrossEntropyLoss loss;
  auto outhpu = loss->forward(houtConv, htarget);
  torch::Tensor out = outhpu.to(torch::kCPU);

  torch::Tensor outConv = torch::conv2d(
      input_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  auto outcpu = loss->forward(outConv, target);

  EXPECT_EQ(allclose(out, outcpu, 0.001, 0.001), true);
}

TEST_F(LazySoftmaxKernelTest, LogSoftMaxTestBackward) {
  torch::Tensor input = torch::rand({64, 10}, torch::requires_grad(false));
  torch::Tensor grad = torch::rand({64, 10}, torch::requires_grad(false));
  torch::Tensor output = torch::rand({64, 10}, torch::requires_grad(false));

  torch::Tensor hinput = input.to(torch::kHPU);
  torch::Tensor hgrad = grad.to(torch::kHPU);
  torch::Tensor houtput = output.to(torch::kHPU);

  constexpr int dim = 0;
  auto hout_backward = torch::_log_softmax_backward_data(
      hgrad, houtput, dim, hinput.scalar_type());

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(hout_backward)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto hout2_back = hout_backward.to(torch::kCPU);

  auto cout_back =
      _log_softmax_backward_data(grad, output, dim, input.scalar_type());

  EXPECT_EQ(allclose(hout2_back, cout_back), true);
}

TEST_F(LazySoftmaxKernelTest, SoftMaxTest) {
  torch::Tensor input = torch::rand({64, 10}, torch::requires_grad(false));
  torch::Tensor hinput = input.to(torch::kHPU);
  constexpr int dim = 0;
  torch::Tensor hout = torch::_softmax(hinput, dim, false);

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(hout)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto hout1 = hout.to(torch::kCPU);

  auto cout = torch::_softmax(input, dim, false);

  EXPECT_EQ(allclose(hout1, cout), true);
}

TEST_F(LazySoftmaxKernelTest, SoftMaxTestBackward) {
  torch::Tensor input = torch::rand({64, 10}, torch::requires_grad(false));
  torch::Tensor grad = torch::rand({64, 10}, torch::requires_grad(false));
  torch::Tensor output = torch::rand({64, 10}, torch::requires_grad(false));

  torch::Tensor hinput = input.to(torch::kHPU);
  torch::Tensor hgrad = grad.to(torch::kHPU);
  torch::Tensor houtput = output.to(torch::kHPU);

  constexpr int dim = 0;
  auto hout_backward =
      torch::_softmax_backward_data(hgrad, houtput, dim, hinput.scalar_type());

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(hout_backward)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto hout2_back = hout_backward.to(torch::kCPU);

  auto cout_back =
      _softmax_backward_data(grad, output, dim, input.scalar_type());

  EXPECT_EQ(allclose(hout2_back, cout_back), true);
}

TEST_F(LazySoftmaxKernelTest, SoftMaxTestBackward1) {
  torch::Tensor input =
      torch::rand({32, 64, 24, 20}, torch::requires_grad(false));
  torch::Tensor hinput = input.to(torch::kHPU);

  torch::Tensor weight_tensor =
      torch::rand({4, 64, 1, 1}, torch::requires_grad(false));
  torch::Tensor tHabanaW = weight_tensor.to(torch::kHPU);

  torch::Tensor output =
      torch::rand({32, 4, 24, 20}, torch::requires_grad(false));
  torch::Tensor houtput = output.to(torch::kHPU);

  constexpr int dim = 1;
  torch::Tensor houtConv =
      torch::conv2d(hinput, tHabanaW, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  auto hout_backward = torch::_softmax_backward_data(
      houtConv, houtput, dim, hinput.scalar_type());
  auto hout2_back = hout_backward.to(torch::kCPU);

  torch::Tensor outConv =
      torch::conv2d(input, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  auto cout_back =
      torch::_softmax_backward_data(outConv, output, dim, input.scalar_type());

  EXPECT_EQ(allclose(hout2_back, cout_back, 0.01, 0.01), true);
}

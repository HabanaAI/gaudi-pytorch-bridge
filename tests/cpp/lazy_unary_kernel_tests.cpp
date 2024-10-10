/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;
using namespace at;

class LazyUnaryKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyUnaryKernelTest, ThresholdBackward) {
  auto threshold_backward_test = [](float threshold) {
    auto grad = torch::randn({2, 2}, torch::requires_grad(false));
    auto self = torch::randn({2, 2}, torch::requires_grad(false));

    Scalar scal_value(threshold);

    auto hgrad = grad.to(torch::kHPU);
    auto hself = self.to(torch::kHPU);

    auto hresult = at::threshold_backward(hgrad, hself, scal_value);

    std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(hresult)};
    HbLazyTensor::SyncTensorsGraph(&tensors);

    auto hout = hresult.to(torch::kCPU);
    auto cout = at::threshold_backward(grad, self, scal_value);

    EXPECT_TRUE(allclose(hout, cout));
  };

  threshold_backward_test(0.0);
  threshold_backward_test(0.5);
  threshold_backward_test(-0.5);
  threshold_backward_test(100);
  threshold_backward_test(-100);
}

TEST_F(LazyUnaryKernelTest, ReluInplaceTest) {
  torch::Tensor a = torch::randn({4, 5});
  auto ha = a.to(torch::kHPU);

  auto A = a.clone();
  auto hA = ha.clone();

  A.relu_();
  hA.relu_();

  EXPECT_TRUE(allclose(hA.to("cpu"), A));
}

TEST_F(LazyUnaryKernelTest, Elu) {
  auto A = torch::randn({4, 5});
  auto hA = A.to(torch::kHPU);

  constexpr double alpha = 0.43;

  A = torch::elu(A, alpha);
  hA = torch::elu(hA, alpha);

  EXPECT_TRUE(allclose(A, hA.to("cpu"))) << A << hA.to("cpu");
}

TEST_F(LazyUnaryKernelTest, LeakyReluInplaceTest) {
  auto A = torch::randn({4, 5});
  auto ha = A.to(torch::kHPU);

  auto hA = ha.clone();
  torch::leaky_relu_(A);
  torch::leaky_relu_(hA);

  EXPECT_TRUE(allclose(A, hA.to("cpu")));
}

TEST_F(LazyUnaryKernelTest, LeakyReluTest) {
  auto A = torch::randn({5, 7, 4});
  auto hA = A.to(torch::kHPU);

  auto expectedOutput = torch::leaky_relu(A);
  auto habanaOutput = torch::leaky_relu(hA);

  EXPECT_TRUE(allclose(expectedOutput, habanaOutput.to("cpu")));
}

TEST_F(LazyUnaryKernelTest, LeakyReluBackwardTest) {
  const std::vector<int64_t> dimentions{2, 3};

  auto grad = torch::randn(dimentions, torch::requires_grad(false));
  auto A = torch::randn(dimentions, torch::requires_grad(false));

  Scalar scal_value(0.1);

  auto hgrad = grad.to(torch::kHPU);
  auto hA = A.to(torch::kHPU);

  auto expectedOutput = torch::leaky_relu_backward(grad, A, scal_value, false);
  auto habanaOutput = torch::leaky_relu_backward(hgrad, hA, scal_value, false);

  EXPECT_TRUE(allclose(expectedOutput, habanaOutput.to("cpu")));
}

TEST_F(LazyUnaryKernelTest, FloorInplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({4, 5});

  auto hA = A.to(torch::kHPU);
  A = A.floor_();
  auto exp = torch::floor(A);

  hA = hA.floor_();
  auto result = torch::floor(hA);

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp), true);
}

TEST_F(LazyUnaryKernelTest, FloorTest) {
  auto input_tensor = torch::randn({4, 5});
  torch::Tensor cpu_out = torch::floor(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::floor(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out), true);
}

TEST_F(LazyUnaryKernelTest, LogInplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::range(1, 100, 0.1);

  auto hA = A.to(torch::kHPU);
  A = A.log_();
  auto exp = torch::log_(A);

  hA = hA.log_();
  auto result = torch::log_(hA);

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp), true);
}

TEST_F(LazyUnaryKernelTest, LogTest) {
  auto input_tensor = torch::range(1, 100, 0.1);
  torch::Tensor cpu_out = torch::log(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::log(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out), true);
}

TEST_F(LazyUnaryKernelTest, Log2InplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::range(1, 100, 0.1);

  auto hA = A.to(torch::kHPU);
  A = A.log2_();
  auto exp = torch::log2_(A);

  hA = hA.log2_();
  auto result = torch::log2_(hA);

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp), true);
}

TEST_F(LazyUnaryKernelTest, Log2Test) {
  auto input_tensor = torch::range(1, 100, 0.1);
  torch::Tensor cpu_out = torch::log2(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::log2(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out), true);
}

TEST_F(LazyUnaryKernelTest, SigmoidFwdTest) {
  auto input_tensor =
      torch::arange(4, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 2, 2});
  torch::Tensor cpu_out = torch::sigmoid(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::sigmoid(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out), true);
}

TEST_F(LazyUnaryKernelTest, SigmoidBwdTest) {
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

TEST_F(LazyUnaryKernelTest, ReciprocalTest) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::reciprocal(hA);
  torch::Tensor Out = torch::reciprocal(A);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyUnaryKernelTest, ReciprocalInplaceTest) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::reciprocal_(hA);
  torch::reciprocal_(A);

  EXPECT_EQ(allclose(hA.to(torch::kCPU), A), true);
}

TEST_F(LazyUnaryKernelTest, ReciprocalOut) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);

  torch::Tensor hOut = at::empty_like(hA);
  torch::Tensor Out = at::empty_like(A);

  torch::reciprocal_outf(hA, hOut);
  torch::reciprocal_outf(A, Out);

  EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out));
}

TEST_F(LazyUnaryKernelTest, HardsigmoidTest) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);

  torch::Tensor hOut = torch::hardsigmoid(hA);
  torch::Tensor Out = torch::hardsigmoid(A);
  EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out));
}

TEST_F(LazyUnaryKernelTest, HardsigmoidBwdTest) {
  auto grad = torch::randn({2, 2}, torch::requires_grad(false));
  auto self = torch::randn({2, 2}, torch::requires_grad(false));

  auto hgrad = grad.to(torch::kHPU);
  auto hself = self.to(torch::kHPU);

  torch::Tensor hOut = torch::hardsigmoid_backward(hgrad, hself);
  torch::Tensor Out = torch::hardsigmoid_backward(grad, self);

  EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out));
}

TEST_F(LazyUnaryKernelTest, HardsigmoidInplaceTest) {
  auto A = torch::randn({4, 5});
  auto hA = A.to(torch::kHPU);

  torch::hardsigmoid_(A);
  torch::hardsigmoid_(hA);

  EXPECT_TRUE(allclose(A, hA.to("cpu")));
}

TEST_F(LazyUnaryKernelTest, SqrtTest) {
  auto input_tensor = torch::randn({4, 5});
  torch::Tensor cpu_out = torch::sqrt(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor temp1 = torch::sqrt(tHabanaX);
  auto temp2 = torch::zeros({4, 5}).to(torch::kHPU);
  auto outHabana = torch::add(temp1, temp2);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(
      allclose(hout_lazy, cpu_out, 0.001, 0.001, /*equal_nan*/ true), true);
}

TEST_F(LazyUnaryKernelTest, RoundInplaceTest) {
  torch::Tensor A = torch::randn({4, 5});

  auto hA = A.to(torch::kHPU);
  auto round = torch::round_(A);
  auto result = torch::round_(hA);

  auto out = result.to(kCPU);

  EXPECT_EQ(allclose(out, round), true);
}

TEST_F(LazyUnaryKernelTest, RoundTest) {
  auto input_tensor = torch::randn({4, 5});
  torch::Tensor cpu_out = torch::round(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::round(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out, 0.001, 0.001, true), true);
}

TEST_F(LazyUnaryKernelTest, AbsTest) {
  auto input_tensor = torch::randn({4, 5});
  torch::Tensor cpu_out = torch::abs(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::abs(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out, 0.001, 0.001, true), true);
}

TEST_F(LazyUnaryKernelTest, AbsInplaceTest) {
  torch::Tensor A = torch::randn({4, 5});

  auto hA = A.to(torch::kHPU);
  auto abs = torch::abs_(A);
  auto result = torch::abs_(hA);

  auto out = result.to(kCPU);

  EXPECT_EQ(allclose(out, abs), true);
}

TEST_F(LazyUnaryKernelTest, RsqrtInplaceTest) {
  torch::Tensor A = torch::add(torch::rand({4, 5}), 1);

  auto hA = A.to(torch::kHPU);
  auto rsqrt = torch::rsqrt_(A);
  auto result = torch::rsqrt_(hA);
  auto out = result.to(kCPU);

  EXPECT_EQ(allclose(out, rsqrt), true);
}

TEST_F(LazyUnaryKernelTest, SignTest) {
  auto input_tensor = torch::randn({4, 3});

  torch::Tensor cpu_out = torch::sign(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::sign(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  bool equal = cpu_out.equal(hout_lazy);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyUnaryKernelTest, SignInplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({4, 5});

  auto hA = A.to(torch::kHPU);
  A = A.sign_();
  auto sign = torch::sign(A);

  hA = hA.sign_();
  auto result = torch::sign(hA);

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, sign), true);
}

TEST_F(LazyUnaryKernelTest, SgnTest) {
  auto input_tensor = torch::randn({4, 3});

  torch::Tensor cpu_out = torch::sgn(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::sgn(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  bool equal = cpu_out.equal(hout_lazy);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyUnaryKernelTest, SgnInplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({4, 5});

  auto hA = A.to(torch::kHPU);
  A = A.sgn_();
  auto sgn = torch::sgn(A);

  hA = hA.sign_();
  auto result = torch::sgn(hA);

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, sgn), true);
}

TEST_F(LazyUnaryKernelTest, RsqrtTest) {
  auto input_tensor = torch::add(torch::rand({4, 5}), 1);
  torch::Tensor cpu_out = torch::rsqrt(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::rsqrt(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out, 0.001, 0.001, true), true);
}

TEST_F(LazyUnaryKernelTest, IsfiniteTest) {
  auto input_tensor = torch::Tensor(torch::zeros({5}));
  input_tensor[0] = input_tensor[0] / 0.0;
  input_tensor[1] = 2.0 / 0.0;
  input_tensor[2] = -2.0 / 0.0;

  torch::Tensor cpu_out = torch::isfinite(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::isfinite(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  bool equal = cpu_out.equal(hout_lazy);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyUnaryKernelTest, ExpTest) {
  auto input_tensor = torch::randn({4, 5});
  torch::Tensor cpu_out = torch::exp(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::exp(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out, 0.001, 0.001, true), true);
}

TEST_F(LazyUnaryKernelTest, ErfTest) {
  auto input_tensor = torch::randn({4, 5});
  torch::Tensor cpu_out = torch::erf(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::erf(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out, 0.001, 0.001, true), true);
}

TEST_F(LazyUnaryKernelTest, ErfInplaceTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::dtype(torch::kFloat));

  auto hA = A.to(torch::kHPU);

  auto exp = torch::erf_(A);
  auto result = torch::erf_(hA);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

TEST_F(LazyUnaryKernelTest, ExpInplaceTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::dtype(torch::kFloat));

  auto hA = A.to(torch::kHPU);

  auto exp = torch::exp_(A);
  auto result = torch::exp_(hA);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

TEST_F(LazyUnaryKernelTest, ClampTest) {
  auto input_tensor = torch::randn({8, 24, 24, 3});
  auto hinput = input_tensor.to(torch::kHPU);
  Scalar min_value(-0.25);
  Scalar max_value(0.25);
  torch::Tensor cpu_out = torch::clamp(input_tensor, min_value, max_value);

  torch::Tensor hresult = torch::clamp(hinput, min_value, max_value);
  auto hout = hresult.to(torch::kCPU);

  EXPECT_EQ(allclose(hout, cpu_out, 0.001, 0.001, /*equal_nan*/ true), true);
}

TEST_F(LazyUnaryKernelTest, ClampInPlaceTest1) {
  auto options = torch::TensorOptions().dtype(torch::kInt32);
  auto input_tensor = torch::randint(-100, 100, {9}, options);
  auto hinput = input_tensor.to(torch::kHPU);
  Scalar min_value(0);
  Scalar max_value(19);
  torch::Tensor cpu_out = torch::clamp_(input_tensor, min_value, max_value);

  torch::Tensor hresult = torch::clamp_(hinput, min_value, max_value);
  auto hout = hresult.to(torch::kCPU);
  EXPECT_EQ(allclose(hout, cpu_out, 0.001, 0.001, /*equal_nan*/ true), true);
  EXPECT_EQ(
      allclose(input_tensor, cpu_out, 0.001, 0.001, /*equal_nan*/ true), true);
}

TEST_F(LazyUnaryKernelTest, ClampInPlaceTest) {
  auto input_tensor = torch::randn({8, 24, 24, 3});
  auto hinput = input_tensor.to(torch::kHPU);
  Scalar min_value(-0.25);
  Scalar max_value(0.25);
  torch::Tensor cpu_out = torch::clamp_(input_tensor, min_value, max_value);

  torch::Tensor hresult = torch::clamp_(hinput, min_value, max_value);
  auto hout = hresult.to(torch::kCPU);

  EXPECT_EQ(allclose(hout, cpu_out, 0.001, 0.001, /*equal_nan*/ true), true);
  EXPECT_EQ(
      allclose(input_tensor, cpu_out, 0.001, 0.001, /*equal_nan*/ true), true);
}
TEST_F(LazyUnaryKernelTest, TanhFwdTest) {
  torch::Tensor A =
      torch::arange(6, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 3, 2});
  torch::Tensor out_exp = torch::tanh(A);

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hout = torch::tanh(hA);
  torch::Tensor hout_lazy = hout.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, out_exp), true);
}

TEST_F(LazyUnaryKernelTest, TanhOut) {
  torch::Tensor A =
      torch::arange(6, torch::dtype(torch::kFloat)).reshape({1, 1, 3, 2});
  torch::Tensor hA = A.to(torch::kHPU);

  torch::Tensor out_exp = torch::tanh_outf(A, A);
  torch::Tensor hout = torch::tanh_outf(hA, hA);
  torch::Tensor hout_lazy = hout.to(torch::kCPU);

  EXPECT_TRUE(allclose(hout_lazy, out_exp));
}

TEST_F(LazyUnaryKernelTest, TanhBwdTest) {
  torch::Tensor A =
      torch::arange(6, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 3, 2});
  auto grad = torch::arange(6, torch::dtype(torch::kFloat).requires_grad(true))
                  .reshape({1, 1, 3, 2});
  torch::Tensor out_exp = torch::tanh_backward(grad, A);

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hGrad = grad.to(torch::kHPU);
  torch::Tensor hout = torch::tanh_backward(hGrad, hA);
  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(hout)};
  HbLazyTensor::SyncTensorsGraph(&tensors);
  auto hout_lazy = hout.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, out_exp), true);
}

TEST_F(LazyUnaryKernelTest, HardTanhTest) {
  auto input_tensor = torch::randn({8, 24, 24, 3});
  auto hinput = input_tensor.to(torch::kHPU);
  Scalar min_value(-0.25);
  Scalar max_value(0.25);
  torch::Tensor cpu_out = torch::hardtanh(input_tensor, min_value, max_value);

  torch::Tensor hresult = torch::hardtanh(hinput, min_value, max_value);
  auto hout = hresult.to(torch::kCPU);

  EXPECT_EQ(allclose(hout, cpu_out, 0.001, 0.001, /*equal_nan*/ true), true);
}

TEST_F(LazyUnaryKernelTest, HardTanhInPlaceTest) {
  auto input_tensor = torch::randn({8, 24, 24, 3});
  auto hinput = input_tensor.to(torch::kHPU);
  Scalar min_value(-0.25);
  Scalar max_value(0.25);
  torch::Tensor cpu_out = torch::hardtanh_(input_tensor, min_value, max_value);

  torch::Tensor hresult = torch::hardtanh_(hinput, min_value, max_value);
  auto hout = hresult.to(torch::kCPU);

  EXPECT_EQ(allclose(hout, cpu_out, 0.001, 0.001, /*equal_nan*/ true), true);
  EXPECT_EQ(
      allclose(input_tensor, cpu_out, 0.001, 0.001, /*equal_nan*/ true), true);
}

TEST_F(LazyUnaryKernelTest, HardTanhInPlaceTest1) {
  auto options = torch::TensorOptions().dtype(torch::kInt32);
  auto input_tensor = torch::randint(-100, 100, {9}, options);
  auto hinput = input_tensor.to(torch::kHPU);
  Scalar min_value(0);
  Scalar max_value(19);
  torch::Tensor cpu_out = torch::hardtanh_(input_tensor, min_value, max_value);

  torch::Tensor hresult = torch::hardtanh_(hinput, min_value, max_value);
  auto hout = hresult.to(torch::kCPU);
  EXPECT_EQ(allclose(hout, cpu_out, 0.001, 0.001, /*equal_nan*/ true), true);
  EXPECT_EQ(
      allclose(input_tensor, cpu_out, 0.001, 0.001, /*equal_nan*/ true), true);
}

TEST_F(LazyUnaryKernelTest, HardTanhBwdTest) {
  torch::Tensor A =
      torch::arange(8, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 4, 2});
  auto grad = torch::arange(8, torch::dtype(torch::kFloat).requires_grad(true))
                  .reshape({1, 1, 4, 2});
  Scalar min_value(0.0);
  Scalar max_value(6.0);
  torch::Tensor out_exp =
      torch::hardtanh_backward(grad, A, min_value, max_value);

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hGrad = grad.to(torch::kHPU);
  torch::Tensor hout =
      torch::hardtanh_backward(hGrad, hA, min_value, max_value);
  auto hout_lazy = hout.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, out_exp), true);
}

TEST_F(LazyUnaryKernelTest, GeluTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::dtype(torch::kFloat));
  auto hA = A.to(torch::kHPU);

  auto exp = torch::nn::functional::gelu(A);
  auto result = torch::nn::functional::gelu(hA);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

TEST_F(LazyUnaryKernelTest, GeluBackward) {
  auto grad = torch::randn({2, 2});
  auto self = torch::randn({2, 2});

  auto hgrad = grad.to(torch::kHPU);
  auto hself = self.to(torch::kHPU);

  auto hresult = at::gelu_backward(hgrad, hself);

  auto hout = hresult.to(torch::kCPU);
  auto cout = at::gelu_backward(grad, self);

  EXPECT_EQ(allclose(hout, cout, 0.001, 0.001), true);
}

TEST_F(LazyUnaryKernelTest, TopkTest) {
  auto self = torch::randn({3, 5});
  auto hself = self.to(torch::kHPU);

  auto out_cpu = at::topk(self, 2, 1, true, true);
  at::Tensor cout = std::get<0>(out_cpu);
  auto out_hpu = at::topk(hself, 2, 1, true, true);
  at::Tensor hout = std::get<0>(out_hpu).to(torch::kCPU);

  EXPECT_EQ(cout.sizes().vec() == hout.sizes().vec(), true);

  EXPECT_EQ(allclose(cout, hout, 0.001, 0.001), true);
}

TEST_F(LazyUnaryKernelTest, TopkOutTest) {
  auto self = torch::randn({3, 5, 4, 7});
  auto hself = self.to(torch::kHPU);

  torch::Tensor out_cpu_values = torch::empty({0});
  torch::Tensor out_cpu_indices = torch::empty({0}, torch::dtype(torch::kLong));

  torch::Tensor out_hpu_values = out_cpu_values.to(torch::kHPU);
  torch::Tensor out_hpu_indices = out_cpu_indices.to(torch::kHPU);

  torch::topk_outf(self, 3, 3, true, true, out_cpu_values, out_cpu_indices);
  at::Tensor cout = out_cpu_values;
  torch::topk_outf(hself, 3, 3, true, true, out_hpu_values, out_hpu_indices);
  at::Tensor hout = out_hpu_values.to(torch::kCPU);

  EXPECT_EQ(cout.sizes().vec() == hout.sizes().vec(), true);

  EXPECT_EQ(allclose(cout, hout, 0.001, 0.001), true);
}

TEST_F(LazyUnaryKernelTest, TopkTestFalse) {
  auto self = torch::randint(0, 1000, {16, 8}, torch::dtype(torch::kInt64));
  auto hself = self.to(torch::kHPU);

  auto out_cpu = at::topk(self, 4, 1, false, true);
  at::Tensor cout = std::get<0>(out_cpu);
  auto out_hpu = at::topk(hself, 4, 1, false, true);
  at::Tensor hout = std::get<0>(out_hpu).to(torch::kCPU);

  EXPECT_EQ(cout.sizes().vec() == hout.sizes().vec(), true);

  EXPECT_EQ(allclose(cout, hout, 0.001, 0.001), true);
}

TEST_F(LazyUnaryKernelTest, SortTest) {
  auto sort_test = [](bool desc, int dim) {
    auto self = torch::randn({3, 5});
    auto hself = self.to(torch::kHPU);

    auto out_cpu = at::sort(self, dim, desc);
    at::Tensor cout = std::get<0>(out_cpu);
    auto out_hpu = at::sort(hself, dim, desc);
    at::Tensor hout = std::get<0>(out_hpu).to(torch::kCPU);

    EXPECT_EQ(cout.sizes().vec() == hout.sizes().vec(), true);
    EXPECT_EQ(allclose(cout, hout, 0.001, 0.001), true);
  };
  sort_test(true, 1);
  sort_test(false, 1);
  sort_test(true, 0);
  sort_test(false, 0);
}

TEST_F(LazyUnaryKernelTest, SortFwdBwdTest) {
  auto N = 8;
  auto C = 1024;
  auto self =
      torch::randn({N, C}, torch::dtype(torch::kFloat).requires_grad(true));
  auto hself = self.to(torch::kHPU).detach();
  hself.requires_grad_(true);

  auto out_cpu = torch::sort(self, 1, false);
  auto cout = std::get<0>(out_cpu);
  auto cout_idx = std::get<1>(out_cpu);
  auto grad_cout = torch::ones_like(cout);
  auto hgrad_cout = grad_cout.to(torch::kHPU).detach();
  hgrad_cout.requires_grad_(true);
  cout.backward(grad_cout);
  auto grad_in = self.grad();

  auto out_hpu = torch::sort(hself, 1, false);
  auto hout = std::get<0>(out_hpu);
  auto hout_cpu = hout.to(torch::kCPU);
  auto hout_idx = std::get<1>(out_hpu).to(torch::kCPU).to(torch::kInt64);
  hout.backward(hgrad_cout);
  auto hgrad_in = hself.grad();
  auto hgrad_in_cpu = hgrad_in.to(torch::kCPU);

  EXPECT_EQ(cout.sizes().vec() == hout_cpu.sizes().vec(), true);
  EXPECT_EQ(cout_idx.sizes().vec() == hout_idx.sizes().vec(), true);

  EXPECT_EQ(
      allclose(cout, hout_cpu, 0.001, 0.001) &&
          allclose(cout_idx, hout_idx, 0.001, 0.001) &&
          allclose(grad_in, hgrad_in_cpu, 0.001, 0.001),
      true);
}

TEST_F(LazyUnaryKernelTest, LeakyReluAutogradZeroSlope) {
  auto device = "hpu:0";
  auto options = torch::TensorOptions().device(device).requires_grad(true);
  auto a = torch::tensor({-2., 0., 2.}, options);
  auto a_clone = a.clone();

  auto b = torch::leaky_relu_(a_clone, 0.0); // check inplace
  b.backward(torch::ones(3, device));

  auto expected = torch::tensor({0., 0., 1.}, device);
  EXPECT_TRUE(allclose(a.grad().to("cpu"), expected.to("cpu")));
}

TEST_F(LazyUnaryKernelTest, NegOut) {
  torch::Tensor A = torch::randn({2, 5});
  torch::Tensor hA = A.to(torch::kHPU);

  torch::Tensor out_exp = at::empty_like(A);
  torch::Tensor hout = at::empty_like(hA);
  torch::neg_outf(A, out_exp);
  torch::neg_outf(hA, hout);

  EXPECT_TRUE(allclose(hout.cpu(), out_exp));
}

TEST_F(LazyUnaryKernelTest, ClampMinTest) {
  auto input_tensor = torch::randn({8, 24, 24, 3});
  auto hinput = input_tensor.to(torch::kHPU);
  Scalar min_value(-0.25);
  torch::Tensor cpu_out = torch::clamp_min(input_tensor, min_value);

  torch::Tensor hresult = torch::clamp_min(hinput, min_value);
  auto hout = hresult.to(torch::kCPU);

  EXPECT_TRUE(allclose(hout, cpu_out, 0.001, 0.001, /*equal_nan*/ true));
}

TEST_F(LazyUnaryKernelTest, Isnan) {
  const std::vector<int64_t> dimentions{3, 3};
  auto input_tensor = torch::randn(dimentions, torch::requires_grad(false));
  input_tensor[1][2] = sqrt(-2); // One element explicitly set to nan
  auto hinput = input_tensor.to(torch::kHPU);

  auto hresult = torch::isnan(hinput);
  auto hout = hresult.to(torch::kCPU);

  auto cpu_out = torch::isnan(input_tensor);
  EXPECT_TRUE(allclose(hout.to(torch::kInt8), cpu_out.to(torch::kInt8)));
}

TEST_F(LazyUnaryKernelTest, SinTest) {
  auto A = torch::randn({4, 5});
  auto hA = A.to(torch::kHPU);

  A = torch::sin(A);
  hA = torch::sin(hA);
  auto hout = hA.to("cpu");

  EXPECT_TRUE(allclose(hout, A));
}

TEST_F(LazyUnaryKernelTest, CosTest) {
  auto A = torch::tensor(1.0);
  auto hA = A.to(torch::kHPU);

  A = torch::cos(A);
  hA = torch::cos(hA);
  auto hout = hA.to("cpu");

  EXPECT_TRUE(allclose(hout, A));
}

TEST_F(LazyUnaryKernelTest, CumsumDim3Axis2) {
  auto A = torch::randn({2, 3, 2}, torch::requires_grad(false));

  auto hA = A.to(torch::kHPU);
  int64_t axis = 2;
  torch::Tensor cpu_out = torch::cumsum(A, axis);

  torch::Tensor hresult = torch::cumsum(hA, axis);
  auto hout = hresult.to(torch::kCPU);
  EXPECT_TRUE(allclose(hout, cpu_out));
}

TEST_F(LazyUnaryKernelTest, CumsumDim2Axis1Int) {
  torch::Tensor A = torch::randint(-330, 330, {2, 3});

  auto hA = A.to(torch::kHPU);
  int64_t axis = 1;
  torch::Tensor cpu_out = torch::cumsum(A, axis, torch::kFloat32);

  torch::Tensor hresult = torch::cumsum(hA, axis, torch::kFloat32);
  auto hout = hresult.to(torch::kCPU);
  EXPECT_TRUE(allclose(hout, cpu_out));
}

TEST_F(LazyUnaryKernelTest, CumsumDim3AxisNe1) {
  auto A = torch::randn({2, 3, 2}, torch::requires_grad(false));

  auto hA = A.to(torch::kHPU);
  int64_t axis = -1;
  torch::Tensor cpu_out = torch::cumsum(A, axis);

  torch::Tensor hresult = torch::cumsum(hA, axis);
  auto hout = hresult.to(torch::kCPU);
  EXPECT_TRUE(allclose(hout, cpu_out));
}

TEST_F(LazyUnaryKernelTest, Cumsum0D) {
  torch::Tensor A = torch::tensor(9.03);
  auto hinput = A.to(torch::kHPU);

  auto hresult = torch::cumsum(hinput, 0, torch::kFloat32);
  auto hout = hresult.to(torch::kCPU);

  auto cpu_out = torch::cumsum(A, 0, torch::kFloat32);
  EXPECT_TRUE(allclose(hout, cpu_out));
}

TEST_F(LazyUnaryKernelTest, EluBackwardTest) {
  const std::vector<int64_t> dimentions{2, 3};

  auto grad = torch::randn(dimentions, torch::requires_grad(false));
  auto A = torch::randn(dimentions, torch::requires_grad(false));

  Scalar alpha(2.0);
  Scalar scale(1.0);
  Scalar input_scale(1.0);
  bool is_result = false;

  auto hgrad = grad.to(torch::kHPU);
  auto hA = A.to(torch::kHPU);

  auto expectedOutput =
      torch::elu_backward(grad, alpha, scale, input_scale, is_result, A);
  auto habanaOutput =
      torch::elu_backward(hgrad, alpha, scale, input_scale, is_result, hA);

  EXPECT_TRUE(allclose(
      expectedOutput,
      habanaOutput.to("cpu"),
      0.001,
      0.001,
      /*equal_nan*/ true));
}

// Also Validates InferOutputMeta for GUID cast_u8_to_f32, cast_f32_to_i32,
// cast_i32_to_bf16
TEST_F(LazyUnaryKernelTest, CastU8F32I32) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  torch::Tensor A = torch::randint(
      0, 255, {2, 3, 4}, torch::dtype(torch::kUInt8).requires_grad(false));
  torch::Tensor out_cpu =
      A.to(torch::kFloat).to(torch::kInt32).to(torch::kBFloat16);

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor out_hpu =
      hA.to(torch::kFloat).to(torch::kInt32).to(torch::kBFloat16);

  EXPECT_EQ(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also Validates InferOutputMeta for cast bf16->i32, i32->u8 and u8->bf16
// cast bf16->i32 validates GUIDs cast_bf16_to_f32 and cast_f32_to_i32
// cast i32->u8 validates GUID cast_i32_to_u8
// cast u8->bf16 validates GUIDs cast_u8_to_f32 and cast_f32_to_bf16
TEST_F(LazyUnaryKernelTest, CastBF16I32U8BF16) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  torch::Tensor A = torch::rand(
      {2, 3, 4}, torch::dtype(torch::kBFloat16).requires_grad(false));
  torch::Tensor out_cpu =
      A.to(torch::kInt32).to(torch::kByte).to(torch::kBFloat16);

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor out_hpu =
      hA.to(torch::kInt32).to(torch::kByte).to(torch::kBFloat16);

  EXPECT_EQ(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also Validates InferOutputMeta for cast byte to bool
TEST_F(LazyUnaryKernelTest, CastF32I32ByteBool) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  auto A = torch::rand({2, 3, 4});
  auto out = A.to(torch::kInt).to(torch::kByte).to(torch::kBool);

  // Char cast to Bool inserts Identity Op
  auto hA = A.to(torch::kHPU);
  auto hOut = hA.to(torch::kInt).to(torch::kByte).to(torch::kBool);

  EXPECT_EQ(
      allclose(
          hOut.to(torch::kFloat).to(torch::kCPU),
          out.to(torch::kFloat),
          0.001,
          0.001),
      true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also Validates InferOutputMeta for GUID cast_f32_to_i8, identity,
// cast_i8_to_f32
TEST_F(LazyUnaryKernelTest, CastIdentity) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  auto A = torch::rand({2, 3, 4});
  auto out = A.to(torch::kChar).to(torch::kBool).to(torch::kFloat);

  // Char cast to Bool inserts Identity Op
  auto hA = A.to(torch::kHPU);
  auto hOut = hA.to(torch::kChar).to(torch::kBool).to(torch::kFloat);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), out, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also Validates InferOutputMeta for GUID memcpy
TEST_F(LazyUnaryKernelTest, CopyD2D) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  auto A = torch::rand({2, 3, 4});
  auto ones = torch::ones_like(A);

  auto hA = A.to(torch::kHPU);
  auto hOnes = ones.to(torch::kHPU);

  auto res = torch::add(A, ones);
  auto hRes = torch::add(hA, hOnes);

  auto out = torch::zeros_like(A);
  auto hOut = out.to(torch::kHPU);

  out.copy_(res);
  hOut.copy_(hRes);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), out, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates InferOutputMeta for GUID atan2_fwd_f32
template <bool outMode, class F>
static void TestInfNan(bool ndims, c10::ScalarType dType, F ptFun) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }

  std::vector<int64_t> dimensions;
  if (ndims) {
    dimensions = {2, 3, 2, 3, 2, 3, 2, 3};
  } else {
    dimensions = {4, 5, 3};
  }

  torch::Tensor A;
  switch (dType) {
    case torch::kInt32:
      A = torch::randint(1000, dimensions);
      break;
    case torch::kFloat32:
    case torch::kBFloat16: {
      A = torch::rand(dimensions);

      static const std::array<float, 5> special = {
          0.0, -0.0, INFINITY, -INFINITY, NAN};

      auto ptr = (float*)A.data_ptr();
      for (auto v : special) {
        *ptr++ = v;
      }
    } break;
    default:
      EXPECT_TRUE(false);
  }

  A = A.to(dType);
  auto hA = A.to(torch::kHPU);

  Tensor expected;
  Tensor result;
  if constexpr (outMode) {
    if (ndims) {
      // Maximum supported input/output tensor dimensions for constant_f32
      // is 5. So we can't initialize with zeros_like or anything else
      // similar.
      expected = A;
      result = hA;
    } else {
      expected = torch::zeros_like(A);
      result = torch::zeros_like(hA);
    }
    expected = expected.to(torch::kBool);
    result = result.to(torch::kBool);
    ptFun(expected, A);
    ptFun(result, hA);
  } else {
    expected = ptFun(A);
    result = ptFun(hA);
  }

  Tensor generated = result.to(kCPU);

  double rtol = 1e-03; // NOLINT
  double atol = 1e-03; // NOLINT

  EXPECT_TRUE(at::allclose(expected, generated, rtol, atol, true));
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

#define TEST_CASE(                                            \
    testName,                                                 \
    outFlag,                                                  \
    ndFlag,                                                   \
    torchNode,                                                \
    torchDtype,                                               \
    lambdaParams,                                             \
    lambdaCall)                                               \
  TEST_F(LazyUnaryKernelTest, Is##testName) {                 \
    TestInfNan<outFlag>(ndFlag, torchDtype, [] lambdaParams { \
      return torch::is##torchNode lambdaCall;                 \
    });                                                       \
  }

#define TEST_WITH_ND(                                                   \
    testName, outFlag, torchNode, torchDtype, lambdaParams, lambdaCall) \
  TEST_CASE(                                                            \
      testName,                                                         \
      outFlag,                                                          \
      false,                                                            \
      torchNode,                                                        \
      torchDtype,                                                       \
      lambdaParams,                                                     \
      lambdaCall)                                                       \
  TEST_CASE(                                                            \
      testName##Nd,                                                     \
      outFlag,                                                          \
      true,                                                             \
      torchNode,                                                        \
      torchDtype,                                                       \
      lambdaParams,                                                     \
      lambdaCall)

#define TEST_DTYPES(testName, outFlag, torchNode, lambdaParams, lambdaCall) \
  TEST_WITH_ND(                                                             \
      testName##F32,                                                        \
      outFlag,                                                              \
      torchNode,                                                            \
      torch::kFloat32,                                                      \
      lambdaParams,                                                         \
      lambdaCall)                                                           \
  TEST_WITH_ND(                                                             \
      testName##BF16,                                                       \
      outFlag,                                                              \
      torchNode,                                                            \
      torch::kBFloat16,                                                     \
      lambdaParams,                                                         \
      lambdaCall)                                                           \
  TEST_WITH_ND(                                                             \
      testName##I32,                                                        \
      outFlag,                                                              \
      torchNode,                                                            \
      torch::kInt32,                                                        \
      lambdaParams,                                                         \
      lambdaCall)

#define TEST_(testName, torchNode) \
  TEST_DTYPES(testName, false, torchNode, (auto A), (A))

#define TEST_OUT(testName, torchNode) \
  TEST_DTYPES(                        \
      testName##Out, true, torchNode##_out, (auto out, auto A), (out, A))

#define TEST_WITH_OUT(testName, torchNode) \
  TEST_(testName, torchNode)               \
  TEST_OUT(testName, torchNode)

TEST_(Finite, finite)
TEST_(Nan, nan)
TEST_(Inf, inf)
TEST_OUT(Inf, inf)
TEST_WITH_OUT(Neginf, neginf)
TEST_WITH_OUT(Posinf, posinf)

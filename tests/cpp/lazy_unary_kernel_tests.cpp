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

class LazyUnaryKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyUnaryKernelTest, ThresholdBackward) {
  auto grad = torch::randn({2, 2}, torch::requires_grad(false));
  auto self = torch::randn({2, 2}, torch::requires_grad(false));

  Scalar scal_value(0);

  auto hgrad = grad.to(torch::kHABANA);
  auto hself = self.to(torch::kHABANA);

  auto hresult = at::threshold_backward(hgrad, hself, scal_value);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hresult)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto hout = hresult.to(torch::kCPU);
  auto cout = at::threshold_backward(grad, self, scal_value);

  EXPECT_EQ(allclose(hout, cout), true);
}

TEST_F(LazyUnaryKernelTest, ReluInplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({4, 5});

  auto hA = A.to(torch::kHABANA);
  A = A.relu_();
  auto exp = torch::relu(A);

  hA = hA.relu_();
  auto result = torch::relu(hA);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp), true);
}

TEST_F(LazyUnaryKernelTest, FloorInplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({4, 5});

  auto hA = A.to(torch::kHABANA);
  A = A.floor_();
  auto exp = torch::floor(A);

  hA = hA.floor_();
  auto result = torch::floor(hA);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp), true);
}

TEST_F(LazyUnaryKernelTest, FloorTest) {
  auto input_tensor = torch::randn({4, 5});
  torch::Tensor cpu_out = torch::floor(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  torch::Tensor outHabana = torch::floor(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out), true);
}

TEST_F(LazyUnaryKernelTest, LogInplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::range(1, 100, 0.1);

  auto hA = A.to(torch::kHABANA);
  A = A.log_();
  auto exp = torch::log_(A);

  hA = hA.log_();
  auto result = torch::log_(hA);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp), true);
}

TEST_F(LazyUnaryKernelTest, LogTest) {
  auto input_tensor = torch::range(1, 100, 0.1);
  torch::Tensor cpu_out = torch::log(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  torch::Tensor outHabana = torch::log(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out), true);
}

TEST_F(LazyUnaryKernelTest, Log2InplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::range(1, 100, 0.1);

  auto hA = A.to(torch::kHABANA);
  A = A.log2_();
  auto exp = torch::log2_(A);

  hA = hA.log2_();
  auto result = torch::log2_(hA);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp), true);
}

TEST_F(LazyUnaryKernelTest, Log2Test) {
  auto input_tensor = torch::range(1, 100, 0.1);
  torch::Tensor cpu_out = torch::log2(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  torch::Tensor outHabana = torch::log2(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out), true);
}

TEST_F(LazyUnaryKernelTest, SigmoidFwdTest) {
  auto input_tensor =
      torch::arange(4, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 2, 2});
  torch::Tensor cpu_out = torch::sigmoid(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
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

  torch::Tensor tHabanaI = input_tensor.to(torch::kHABANA);
  torch::Tensor tHabanaG = grad_tensor.to(torch::kHABANA);
  torch::Tensor hout_backward = torch::sigmoid_backward(tHabanaG, tHabanaI);
  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hout_backward)};
  HbLazyTensor::SyncTensorsGraph(&tensors);
  auto hout_lazy = hout_backward.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out), true);
}

TEST_F(LazyUnaryKernelTest, ReciprocalTest) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = torch::reciprocal(hA);
  torch::Tensor Out = torch::reciprocal(A);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyUnaryKernelTest, SqrtTest) {
  auto input_tensor = torch::randn({4, 5});
  torch::Tensor cpu_out = torch::sqrt(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  torch::Tensor temp1 = torch::sqrt(tHabanaX);
  auto temp2 = torch::zeros({4, 5}).to(torch::kHABANA);
  auto outHabana = torch::add(temp1, temp2);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(
      allclose(hout_lazy, cpu_out, 0.001, 0.001, /*equal_nan*/ true), true);
}

TEST_F(LazyUnaryKernelTest, RoundInplaceTest) {
  torch::Tensor A = torch::randn({4, 5});

  auto hA = A.to(torch::kHABANA);
  auto round = torch::round_(A);
  auto result = torch::round_(hA);

  auto out = result.to(kCPU);

  EXPECT_EQ(allclose(out, round), true);
}

TEST_F(LazyUnaryKernelTest, RoundTest) {
  auto input_tensor = torch::randn({4, 5});
  torch::Tensor cpu_out = torch::round(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  torch::Tensor outHabana = torch::round(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out, 0.001, 0.001, true), true);
}

TEST_F(LazyUnaryKernelTest, RsqrtInplaceTest) {
  torch::Tensor A = torch::add(torch::rand({4, 5}), 1);

  auto hA = A.to(torch::kHABANA);
  auto rsqrt = torch::rsqrt_(A);
  auto result = torch::rsqrt_(hA);
  auto out = result.to(kCPU);

  EXPECT_EQ(allclose(out, rsqrt), true);
}

TEST_F(LazyUnaryKernelTest, RsqrtTest) {
  auto input_tensor = torch::add(torch::rand({4, 5}), 1);
  torch::Tensor cpu_out = torch::rsqrt(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
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

  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  torch::Tensor outHabana = torch::isfinite(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  bool equal = cpu_out.equal(hout_lazy);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyUnaryKernelTest, ErfInplaceTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::dtype(torch::kFloat));

  auto hA = A.to(torch::kHABANA);

  auto exp = torch::erf_(A);
  auto result = torch::erf_(hA);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

TEST_F(LazyUnaryKernelTest, ExpInplaceTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::dtype(torch::kFloat));

  auto hA = A.to(torch::kHABANA);

  auto exp = torch::exp_(A);
  auto result = torch::exp_(hA);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}
TEST_F(LazyUnaryKernelTest, ClampInPlaceTest) {
  auto input_tensor = torch::randn({8, 24, 24, 3});
  auto hinput = input_tensor.to(torch::kHABANA);
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

  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hout = torch::tanh(hA);
  torch::Tensor hout_lazy = hout.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, out_exp), true);
}

TEST_F(LazyUnaryKernelTest, TanhBwdTest) {
  torch::Tensor A =
      torch::arange(6, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 3, 2});
  auto grad = torch::arange(6, torch::dtype(torch::kFloat).requires_grad(true))
                  .reshape({1, 1, 3, 2});
  torch::Tensor out_exp = torch::tanh_backward(grad, A);

  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hGrad = grad.to(torch::kHABANA);
  torch::Tensor hout = torch::tanh_backward(hGrad, hA);
  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hout)};
  HbLazyTensor::SyncTensorsGraph(&tensors);
  auto hout_lazy = hout.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, out_exp), true);
}

TEST_F(LazyUnaryKernelTest, DISABLED_GeluTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::dtype(torch::kFloat));
  auto hA = A.to(torch::kHABANA);

  auto exp = torch::nn::functional::gelu(A);
  auto result = torch::nn::functional::gelu(hA);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

TEST_F(LazyUnaryKernelTest, DISABLED_GeluBackward) {
  auto grad = torch::randn({2, 2});
  auto self = torch::randn({2, 2});

  auto hgrad = grad.to(torch::kHABANA);
  auto hself = self.to(torch::kHABANA);

  auto hresult = at::gelu_backward(hgrad, hself);

  auto hout = hresult.to(torch::kCPU);
  auto cout = at::gelu_backward(grad, self);

  EXPECT_EQ(allclose(hout, cout, 0.001, 0.001), true);
}

TEST_F(LazyUnaryKernelTest, TopkTest) {
  auto self = torch::randn({3, 5});
  auto hself = self.to(torch::kHABANA);

  auto out_cpu = at::topk(self, 2, 1, true, true);
  at::Tensor cout = std::get<0>(out_cpu);
  auto out_hpu = at::topk(hself, 2, 1, true, true);
  at::Tensor hout = std::get<0>(out_hpu).to(torch::kCPU);

  EXPECT_EQ(cout.sizes().vec() == hout.sizes().vec(), true);

  EXPECT_EQ(allclose(cout, hout, 0.001, 0.001), true);
}

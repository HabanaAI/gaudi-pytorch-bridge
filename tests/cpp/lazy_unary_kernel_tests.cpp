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
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

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
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp), true);
}

TEST_F(LazyUnaryKernelTest, SigmoidFwdTest) {
  auto input_tensor = torch::arange(4, torch::dtype(torch::kFloat).requires_grad(true))
                          .reshape({1, 1, 2, 2});
  torch::Tensor cpu_out = torch::sigmoid(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  torch::Tensor outHabana = torch::sigmoid(tHabanaX);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out), true);
}

TEST_F(LazyUnaryKernelTest, SigmoidBwdTest) {
  auto input_tensor = torch::arange(4, torch::dtype(torch::kFloat).requires_grad(true))
                          .reshape({1, 1, 2, 2});
  auto grad_tensor = torch::arange(4, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 2, 2});
  torch::Tensor cpu_out = torch::sigmoid_backward(grad_tensor, input_tensor);

  torch::Tensor tHabanaI= input_tensor.to(torch::kHABANA);
  torch::Tensor tHabanaG = grad_tensor.to(torch::kHABANA);
  torch::Tensor hout_backward = torch::sigmoid_backward(tHabanaG, tHabanaI);
  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hout_backward)};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});
  auto hout_lazy = hout_backward.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out), true);
}

TEST_F(LazyUnaryKernelTest, PowTensorScalarTest) {
  auto input_tensor = torch::randn({4, 5});
  torch::Tensor cpu_out = torch::pow(input_tensor, 2.0);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  torch::Tensor outHabana = torch::pow(tHabanaX, 2.0);
  torch::Tensor hout_lazy = outHabana.to(torch::kCPU);

  EXPECT_EQ(allclose(hout_lazy, cpu_out), true);
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
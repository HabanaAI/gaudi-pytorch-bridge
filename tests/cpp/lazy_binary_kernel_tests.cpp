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

#include <cstdlib>

using namespace habana_lazy;

class LazyBinaryKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyBinaryKernelTest, LazyDoATest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor C = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor hC = C.to(torch::kHABANA);
  torch::Tensor I = torch::add(hA, hB);
  torch::Tensor out = torch::add(hC, I);

  torch::Tensor I_cpu = torch::add(A, B);
  torch::Tensor out_cpu = torch::add(C, I_cpu);
  torch::Tensor out_h = out.to(torch::kCPU);
  EXPECT_EQ(allclose(out_h, out_cpu), true);
}

TEST_F(LazyBinaryKernelTest, AddScalarTest) {
  // test case for result = add(tensor, scalar, alpha)
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  Scalar B = 1.0;

  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor out_h = torch::add(hA, B).to(torch::kCPU);
  torch::Tensor out_cpu = torch::add(A, B);

  EXPECT_EQ(allclose(out_h, out_cpu), true);
}

TEST_F(LazyBinaryKernelTest, AddInplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});

  auto hA = A.to(torch::kHABANA);
  A = A.add_(B);
  auto exp = torch::mul(A, C);

  auto hB = B.to(torch::kHABANA);
  auto hC = C.to(torch::kHABANA);
  hA = hA.add_(hB);
  auto result = torch::mul(hA, hC);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp), true);
}
TEST_F(LazyBinaryKernelTest, LazyRsubscalarTest) {
  torch::Tensor input = torch::ones({10, 10});

  auto hinput = input.to(torch::kHABANA);
  auto hrsub = torch::rsub(hinput, 8, 2);
  Tensor hout = hrsub.to(kCPU);

  auto cout = torch::rsub(input, 8, 2);
  EXPECT_EQ(allclose(hout, cout), true);
}

TEST_F(LazyBinaryKernelTest, DivTensorTestWithDivByZero) {
  const std::vector<int64_t> dimentions{5, 3, 4};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);

  // Make sure some elements of B are zero
  int64_t noOfElement = 1;
  size_t index[dimentions.size()];
  for (unsigned int i = 0; i < dimentions.size(); ++i) {
    noOfElement *= dimentions.at(i);
  }

  int64_t noOfZeros = std::rand() % noOfElement;
  for (int64_t i = 0; i < noOfZeros; ++i) {
    for (unsigned dim = 0; dim < dimentions.size(); ++dim) {
      index[dim] = std::rand() % (dimentions.at(dim) - 1);
    }
    B[index[0]][index[1]][index[2]] = 0.0;
  }

  // Compute expected output
  auto expected = torch::div(A, B);

  // Compute actual output
  auto hA = A.to(torch::kHABANA);
  auto hB = B.to(torch::kHABANA);
  auto result = torch::div(hA, hB);
  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);
  Tensor generated = result.to(kCPU);

  // Compare
  EXPECT_EQ(allclose(generated, expected), true);
}

TEST_F(LazyBinaryKernelTest, DivTensorTestByNonZero) {
  const std::vector<int64_t> dimentions{5, 3, 4};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);

  // Make sure no element of B is zero
  int64_t index[dimentions.size()];
  for (index[0] = 0; index[0] < dimentions[0]; ++index[0]) {
    for (index[1] = 0; index[1] < dimentions[1]; ++index[1]) {
      for (index[2] = 0; index[2] < dimentions[2]; ++index[2]) {
        if (std::numeric_limits<float>::epsilon() >=
            abs(0.0 - B[index[0]][index[1]][index[2]]).item<float>()) {
          B[index[0]][index[1]][index[2]] = 1.0;
        } // if(std::numeric_limits<float>::epsilon()
      }
    } // for(index[1]=0;index[1]
  }

  auto expected = torch::div(A, B);

  auto hA = A.to(torch::kHABANA);
  auto hB = B.to(torch::kHABANA);
  auto result = torch::div(hA, hB);
  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);
  Tensor generated = result.to(kCPU);

  EXPECT_EQ(allclose(generated, expected), true);
}

TEST_F(LazyBinaryKernelTest, MulOutScalar) {
  torch::Tensor input1 = torch::randn({2, 2});
  int divFactor_ = 2;
  auto wrapped = c10::scalar_to_tensor(double(1.) / divFactor_);
  wrapped.unsafeGetTensorImpl()->set_wrapped_number(true);
  torch::Tensor out_cpu = torch::zeros_like(input1);
  torch::Tensor out_hpu = torch::zeros_like(input1).to(torch::kHABANA);
  at::mul_out(out_cpu, input1, wrapped);
  at::mul_out(out_hpu, input1.to(torch::kHABANA), wrapped);
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBinaryKernelTest, MulOut) {
  torch::Tensor input1 = torch::randn({2, 2});
  torch::Tensor input2 = torch::randn({2, 2});
  torch::Tensor out_cpu = torch::zeros_like(input1);
  torch::Tensor out_hpu = torch::zeros_like(input1).to(torch::kHABANA);
  at::mul_out(out_cpu, input1, input2);
  at::mul_out(out_hpu, input1.to(torch::kHABANA), input2.to(torch::kHABANA));
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBinaryKernelTest, MulOutNarrow) {
  torch::Tensor input1 =
      torch::arange(6, torch::dtype(torch::kFloat)).reshape({2, 3});
  torch::Tensor input2 =
      torch::arange(6, torch::dtype(torch::kFloat)).reshape({2, 3});

  torch::Tensor A =
      torch::arange(6, torch::dtype(torch::kFloat)).reshape({2, 3});
  torch::Tensor hA = A.to(torch::kHABANA);
  Tensor out_cpu = A.as_strided({2, 3}, input2.strides(), 0);
  Tensor out_hpu = hA.as_strided({2, 3}, input2.strides(), 0);

  at::mul_out(out_cpu, input1, input2);
  at::mul_out(out_hpu, input1.to(torch::kHABANA), input2.to(torch::kHABANA));
  HbLazyTensor::StepMarker({});
  bool equal = A.allclose(hA.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}
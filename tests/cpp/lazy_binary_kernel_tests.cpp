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

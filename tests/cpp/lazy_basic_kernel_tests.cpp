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

class LazyBasicKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyBasicKernelTest, BasicThreadSafety) {
  torch::Tensor A = torch::rand({20});
  torch::Tensor hA = A.to("habana");

  auto t = std::thread([&]() {
    torch::Tensor g = torch::ones({5});
    torch::Tensor hg = g.to("habana");
    Tensor Out = A.narrow(0, 2, 5);
    Tensor hOut = hA.narrow(0, 2, 5);
    Out.copy_(g.view({-1}), true);
    hOut.copy_(hg.view({-1}), true);
    HbLazyTensor::StepMarker({});
  });

  auto t2 = std::thread([&]() {
    torch::Tensor hg2, g2 = torch::zeros({5});
    hg2 = g2.to("habana");
    Tensor Out2 = A.narrow(0, 8, 5);
    Tensor hOut2 = hA.narrow(0, 8, 5);
    Out2.copy_(g2.view({-1}), true);
    hOut2.copy_(hg2.view({-1}), true);
    HbLazyTensor::StepMarker({});
  });
  t.join();
  t2.join();

  A = A.div_(2);
  hA = hA.div_(2);

  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(A, hA.to("cpu")), true) << A << hA.to("cpu");
}

TEST_F(LazyBasicKernelTest, DoubleCopyTest) {
  at::TensorOptions opts =
      at::TensorOptions().dtype(c10::ScalarType::Double).requires_grad(false);
  torch::Tensor A = torch::randn({50, 50}, opts);
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hA_cpu = hA.to(torch::kCPU);
  // This should be double
  bool equal = hA_cpu.allclose(A, 0.1, 0.1);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBasicKernelTest, BasicCopyTest) {
  at::TensorOptions opts = at::TensorOptions().requires_grad(false);
  torch::Tensor A = torch::randn({50, 50}, opts);
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hA_cpu = hA.to(torch::kCPU);
  bool equal = hA_cpu.allclose(A, 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBasicKernelTest, CloneTest) {
  at::TensorOptions opts = at::TensorOptions().requires_grad(false);
  torch::Tensor A = torch::randn({50, 50}, opts);
  torch::Tensor B = torch::randn({50, 50}, opts);
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor hC = hA + hB;
  torch::Tensor hD = torch::clone(hC);
  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(hD));
  std::vector<HbLazyTensor> tensors = {*hl_result};
  HbLazyTensor::SyncTensorsGraph(&tensors);
  torch::Tensor hC_cpu = hC.to(torch::kCPU);
  torch::Tensor hd_cpu = hD.to(torch::kCPU);
  bool equal = hC_cpu.allclose(hd_cpu, 0, 0);
  EXPECT_EQ(equal, true);
}
TEST_F(LazyBasicKernelTest, ViewCopy) {
  torch::Tensor A = torch::randn({20});
  torch::Tensor hA = A.to(torch::kHABANA);
  Tensor Out = A.narrow(0, 2, 5);
  Tensor hOut = hA.narrow(0, 2, 5);
  torch::Tensor g = torch::ones({5});
  torch::Tensor hg = g.to(torch::kHABANA);
  Out.copy_(g.view({-1}), true);
  hOut.copy_(hg.view({-1}), true);

  Tensor Out2 = A.narrow(0, 8, 5);
  Tensor hOut2 = hA.narrow(0, 8, 5);
  torch::Tensor g2 = torch::zeros({5});
  torch::Tensor hg2 = g2.to(torch::kHABANA);
  Out2.copy_(g2.view({-1}), true);
  hOut2.copy_(hg2.view({-1}), true);
  HbLazyTensor::StepMarker({});
  A = A.div_(2);
  hA = hA.div_(2);
  HbLazyTensor::StepMarker({});
  std::cout << A << "\n";
  std::cout << hA.to(kCPU) << "\n";
  EXPECT_EQ(allclose(hA.to(torch::kCPU), A), true);
}
TEST_F(LazyBasicKernelTest, ControlEdge) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});
  torch::Tensor F = torch::randn({2, 3});
  auto hA = A.to(torch::kHABANA);
  auto hB = B.to(torch::kHABANA);
  auto hC = C.to(torch::kHABANA);
  auto hF = F.to(torch::kHABANA);

  auto D = A.mul(B);
  auto E = C.mul(B);
  B = B.add_(F);

  auto hD = hA.mul(hB);
  auto hE = hC.mul(hB);
  hB = hB.add_(hF);
  HbLazyTensor::StepMarker({});
  Tensor out = hB.to(kCPU);

  EXPECT_EQ(allclose(out, B), true);
}
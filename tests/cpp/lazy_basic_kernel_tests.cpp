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

class LazyBasicKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyBasicKernelTest, DISABLED_BasicThreadSafety) {
  torch::Tensor A = torch::rand({20});
  torch::Tensor hA = A.to("hpu");

  auto t = std::thread([&]() {
    torch::Tensor g = torch::ones({5});
    torch::Tensor hg = g.to("hpu");
    Tensor Out = A.narrow(0, 2, 5);
    Tensor hOut = hA.narrow(0, 2, 5);
    Out.copy_(g.view({-1}), true);
    hOut.copy_(hg.view({-1}), true);
    HbLazyTensor::StepMarker({});
  });

  auto t2 = std::thread([&]() {
    torch::Tensor hg2, g2 = torch::zeros({5});
    hg2 = g2.to("hpu");
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
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hA_cpu = hA.to(torch::kCPU);
  // This should be double
  bool equal = hA_cpu.allclose(A, 0.1, 0.1);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBasicKernelTest, BasicCopyTest) {
  at::TensorOptions opts = at::TensorOptions().requires_grad(false);
  torch::Tensor A = torch::randn({50, 50}, opts);
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hA_cpu = hA.to(torch::kCPU);
  bool equal = hA_cpu.allclose(A, 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBasicKernelTest, CloneTest) {
  at::TensorOptions opts = at::TensorOptions().requires_grad(false);
  torch::Tensor A = torch::randn({50, 50}, opts);
  torch::Tensor B = torch::randn({50, 50}, opts);
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
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
TEST_F(LazyBasicKernelTest, DISABLED_ViewCopy) {
  setenv("PT_HPU_LOWER_AS_STRIDED", "1", 1);
  torch::Tensor A = torch::randn({20});
  torch::Tensor hA = A.to(torch::kHPU);
  Tensor Out = A.narrow(0, 2, 5);
  Tensor hOut = hA.narrow(0, 2, 5);
  torch::Tensor g = torch::ones({5});
  torch::Tensor hg = g.to(torch::kHPU);
  Out.copy_(g.view({-1}), true);
  hOut.copy_(hg.view({-1}), true);

  Tensor Out2 = A.narrow(0, 8, 5);
  Tensor hOut2 = hA.narrow(0, 8, 5);
  torch::Tensor g2 = torch::zeros({5});
  torch::Tensor hg2 = g2.to(torch::kHPU);
  Out2.copy_(g2.view({-1}), true);
  hOut2.copy_(hg2.view({-1}), true);
  HbLazyTensor::StepMarker({});
  A = A.div_(2);
  hA = hA.div_(2);
  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(hA.to(torch::kCPU), A), true);
  unsetenv("PT_HPU_LOWER_AS_STRIDED");
}

TEST_F(LazyBasicKernelTest, NarrowInplaceOffsets) {
  setenv("PT_HPU_LOWER_AS_STRIDED", "1", 1);
  torch::Tensor A = torch::randn({20});
  torch::Tensor hA = A.to(torch::kHPU);

  // cpu
  auto temp1 = A.narrow(0, 2, 5);
  auto temp2 = A.narrow(0, 7, 11);

  auto out1 = temp1.fill_(1.0);
  auto out2 = temp2.fill_(2.0);

  // hpu
  auto htemp1 = hA.narrow(0, 2, 5);
  auto htemp2 = hA.narrow(0, 7, 11);

  auto hout1 = htemp1.fill_(1.0);
  auto hout2 = htemp2.fill_(2.0);

  HbLazyTensor::StepMarker({});

  EXPECT_EQ(allclose(hA.cpu(), A), true);
  unsetenv("PT_HPU_LOWER_AS_STRIDED");
}

TEST_F(LazyBasicKernelTest, ControlEdge) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});
  torch::Tensor F = torch::randn({2, 3});
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);
  auto hF = F.to(torch::kHPU);

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
TEST_F(LazyBasicKernelTest, asStridedOnlyGraph) {
  setenv("PT_HPU_LOWER_AS_STRIDED", "1", 1);
  torch::Tensor A = torch::randn({16});
  auto hA = A.to(torch::kHPU);
  std::vector<int64_t> sz{4};
  std::vector<int64_t> str{1};
  c10::IntArrayRef sizes(sz.data(), sz.size());
  c10::IntArrayRef strides(str.data(), str.size());
  int64_t offset = 0;
  auto hB = as_strided_hpu_lazy(hA, sizes, strides, offset);
  Tensor out = hB.to(kCPU);
  unsetenv("PT_HPU_LOWER_AS_STRIDED");
}

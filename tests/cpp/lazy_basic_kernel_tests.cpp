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
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(LazyBasicKernelTest, DoubleCopyTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  at::TensorOptions opts =
      at::TensorOptions().dtype(c10::ScalarType::Double).requires_grad(false);
  torch::Tensor A = torch::randn({50, 50}, opts);
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hA_cpu = hA.to(torch::kCPU);
  // This should be double
  bool equal = hA_cpu.allclose(A, 0.1, 0.1);
  EXPECT_EQ(equal, true);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyBasicKernelTest, BasicCopyTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  at::TensorOptions opts = at::TensorOptions().requires_grad(false);
  torch::Tensor A = torch::randn({50, 50}, opts);
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hA_cpu = hA.to(torch::kCPU);
  bool equal = hA_cpu.allclose(A, 0, 0);
  EXPECT_EQ(equal, true);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyBasicKernelTest, CloneTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  at::TensorOptions opts = at::TensorOptions().requires_grad(false);
  torch::Tensor A = torch::randn({50, 50}, opts);
  torch::Tensor B = torch::randn({50, 50}, opts);
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor hC = hA + hB;
  torch::Tensor hD = torch::clone(hC);
  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(hD));
  std::vector<HbLazyTensor> tensors = {*hl_result};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});
  torch::Tensor hC_cpu = hC.to(torch::kCPU);
  torch::Tensor hd_cpu = hD.to(torch::kCPU);
  bool equal = hC_cpu.allclose(hd_cpu, 0, 0);
  EXPECT_EQ(equal, true);
  unsetenv("PT_HPU_LAZY_MODE");
}

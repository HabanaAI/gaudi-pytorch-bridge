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

class LazyReductionKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST(LazyReductionKernelTest, SumTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = torch::sum(hA);
  torch::Tensor Out = torch::sum(A);

  std::vector<HbLazyTensor> hl_tensors = {GetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors, {});

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST(LazyReductionKernelTest, SumDimIntTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = torch::sum(hA, 1);
  torch::Tensor Out = torch::sum(A, 1);

  std::vector<HbLazyTensor> hl_tensors = {GetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors, {});

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
  unsetenv("PT_HPU_LAZY_MODE");
}
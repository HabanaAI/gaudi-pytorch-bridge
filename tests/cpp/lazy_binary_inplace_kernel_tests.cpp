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

class LazyBinaryInplaceKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(LazyBinaryInplaceKernelTest, MulInplaceTest) {
  // Inplace op as output node is not supported yet.
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});
  auto hA = A.to(torch::kHABANA);
  auto hB = B.to(torch::kHABANA);
  auto hC = C.to(torch::kHABANA);

  A = A.mul_(B);
  auto exp = torch::add(A, C);

  hA = hA.mul_(hB);
  auto result = torch::add(hA, hC);
  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp), true);
  unsetenv("PT_HPU_LAZY_MODE");
}
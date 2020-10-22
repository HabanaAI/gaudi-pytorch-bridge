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
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(LazyBinaryKernelTest, LazyDoATest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor C = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor hC = C.to(torch::kHABANA);
  torch::Tensor I = torch::add(hA, hB);
  torch::Tensor out = torch::add(hC, I);
  EXPECT_EQ(out.dim(), 2);
  unsetenv("PT_HPU_LAZY_MODE");
}


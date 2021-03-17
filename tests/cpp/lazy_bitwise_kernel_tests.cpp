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

class LazyBitwiseKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyBitwiseKernelTest, BitwiseAddTest) {
  torch::Tensor A = torch::randint(-10, 10, {3, 2}) > 0;
  torch::Tensor B = torch::randint(-10, 10, {3, 2}) > 0;
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor out = torch::bitwise_and(hA, hB);

  torch::Tensor out_cpu = torch::bitwise_and(A, B);
  torch::Tensor out_h = out.to(torch::kCPU);
  EXPECT_EQ(allclose(out_h.to(torch::kI8), out_cpu.to(torch::kI8)), true);
}

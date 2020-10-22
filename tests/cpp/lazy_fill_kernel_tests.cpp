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

class LazyFillKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(LazyFillKernelTest, LocalScalarDenseTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({1}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);

  auto hl_result = GetOrCreateHbLazyTensor(A, A.device());

  // .item() invokes local scalar dense
  auto s = hA.item();
  auto s_cpu = A.item();

  EXPECT_EQ(s.to<float>(), s_cpu.to<float>());

  unsetenv("PT_HPU_LAZY_MODE");
}


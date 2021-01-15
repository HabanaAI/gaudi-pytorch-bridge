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

class LazyRandomGenKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyRandomGenKernelTest, FusedDropoutTest) {
  auto in = torch::randn({64, 4, 28, 28}, torch::dtype(torch::kFloat)); // nchw
  double p = 0.3;

  // auto exp = torch::_fused_dropout(in, p);
  // [TO VERIFY] C++ exception with description "Could not run
  // 'aten::_fused_dropout' with arguments from the 'CPUTensorId' backend.
  // 'aten::_fused_dropout' is only available for these backends:
  // [UNKNOWN_TENSOR_TYPE_ID, VariableTensorId]. (reportError at
  // ../aten/src/ATen/core/dispatch/Dispatcher.cpp:176)

  auto h_in = in.to(torch::kHABANA);
  auto result = torch::_fused_dropout(h_in, p);

  // auto out = result.to(kCPU);
  // EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
}
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

class LazyUnaryKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(LazyUnaryKernelTest, ThresholdBackward) {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
    auto grad = torch::randn({2, 2}, torch::requires_grad(false));
    auto self = torch::randn({2, 2}, torch::requires_grad(false));

    Scalar scal_value(0);

    auto hgrad = grad.to(torch::kHABANA);
    auto hself = self.to(torch::kHABANA);

    auto hresult = at::threshold_backward(hgrad, hself, scal_value);

    std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hresult)};
    HbLazyTensor::SyncTensorsGraph(&tensors, {});

    auto hout = hresult.to(torch::kCPU);
    auto cout = at::threshold_backward(grad, self, scal_value);

    EXPECT_EQ(allclose(hout, cout), true);
    unsetenv("PT_HPU_LAZY_MODE");
}

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

class LazyLossKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyLossKernelTest, MseLossTest) {
  torch::Tensor input = torch::randn({3, 5});
  torch::Tensor target = torch::randn({3, 5});
  torch::Tensor grad_input = torch::randn({3, 5});

  auto hinput = input.to(torch::kHABANA);
  auto htarget = target.to(torch::kHABANA);
  auto hgrad_input = grad_input.to(torch::kHABANA);
  torch::Tensor hout1 = torch::mse_loss(hinput, htarget, at::Reduction::None);
  torch::Tensor hout2 = torch::mse_loss_backward(
      hgrad_input, hinput, htarget, at::Reduction::None);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hout1),
                                       GetHbLazyTensor(hout2)};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

  auto out1 = hout1.to(torch::kCPU);
  auto out2 = hout2.to(torch::kCPU);

  auto exp1 = mse_loss(input, target, at::Reduction::None);
  auto exp2 = mse_loss_backward(grad_input, input, target, at::Reduction::None);

  EXPECT_EQ(allclose(out1, exp1), true);
  EXPECT_EQ(allclose(out2, exp2), true);
}

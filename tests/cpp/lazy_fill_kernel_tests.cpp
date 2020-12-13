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
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyFillKernelTest, LocalScalarDenseTest) {
  torch::Tensor A = torch::randn({1}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);

  // .item() invokes local scalar dense
  auto s = hA.item();
  auto s_cpu = A.item();

  EXPECT_EQ(s.to<float>(), s_cpu.to<float>());
}

TEST_F(LazyFillKernelTest, ExecuteFillGraph) {
  Tensor tensor_in1 = torch::randn({2});
  torch::Tensor htensor_in1 = tensor_in1.to(torch::kHABANA);
  auto out = htensor_in1.fill_(1.0);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(out)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto exp = tensor_in1.fill_(1.0);
  auto out_cpu = htensor_in1.to(torch::kCPU);
  EXPECT_EQ(allclose(out_cpu, exp), true);
}

TEST_F(LazyFillKernelTest, ExecuteZerosGraph) {
  Tensor tensor_in1 = torch::randn({2});
  torch::Tensor htensor_in1 = tensor_in1.to(torch::kHABANA);

  auto out = htensor_in1.zero_();

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(out)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto exp = tensor_in1.zero_();
  auto out_cpu = out.to(torch::kCPU);

  EXPECT_EQ(allclose(out_cpu, exp), true);
}

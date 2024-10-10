#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;
using namespace at;

class LazyFillKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyFillKernelTest, LocalScalarDenseTest) {
  torch::Tensor A = torch::randn({1}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);

  // .item() invokes local scalar dense
  auto s = hA.item();
  auto s_cpu = A.item();

  EXPECT_EQ(s.to<float>(), s_cpu.to<float>());
}

TEST_F(LazyFillKernelTest, ExecuteFillGraph) {
  Tensor tensor_in1 = torch::randn({2});
  torch::Tensor htensor_in1 = tensor_in1.to(torch::kHPU);
  auto out = htensor_in1.fill_(1.0);

  auto exp = tensor_in1.fill_(1.0);
  auto out_cpu = htensor_in1.to(torch::kCPU);
  EXPECT_EQ(allclose(out_cpu, exp), true);
}

TEST_F(LazyFillKernelTest, ExecuteZerosGraph) {
  Tensor tensor_in1 = torch::randn({2});
  torch::Tensor htensor_in1 = tensor_in1.to(torch::kHPU);

  auto out = htensor_in1.zero_();

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(out)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto exp = tensor_in1.zero_();
  auto out_cpu = out.to(torch::kCPU);

  EXPECT_EQ(allclose(out_cpu, exp), true);
}

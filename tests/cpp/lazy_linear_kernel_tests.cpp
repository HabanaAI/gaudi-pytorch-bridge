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

class LazyLinearKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(LazyLinearKernelTest, MmMulTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto x = torch::randn({2, 3});
  auto y = torch::randn({3, 3});
  auto z = torch::randn({2, 3});
  torch::Tensor hx = x.to(torch::kHABANA);
  torch::Tensor hy = y.to(torch::kHABANA);
  torch::Tensor hz = z.to(torch::kHABANA);

  auto hy_exp = torch::mm(hx, hy);
  auto hz_exp = torch::mul(hy_exp, hz);
  // Match lazy IR graph
  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(hz_exp));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};
  auto out_string = IrGraphDumpUtil::ToText(a);

  EXPECT_EQ(
      out_string.find("IR {\n"
                      "  %0 = hpu::input()\n"
                      "  %1 = hpu::input()\n"
                      "  %2 = hpu::input()\n"
                      "  %3 = aten::mm(%2, %1)\n"
                      "  %4 = aten::mul(%3, %0), ROOT=0\n"
                      "}"),
      0);

  // Match expectd output
  // ASSERT_TRUE(torch::allclose(hz_exp, hz_exp));
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyLinearKernelTest, AddMmTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2});
  torch::Tensor B = torch::randn({2, 2});
  torch::Tensor C = torch::randn({2, 2});

  torch::Tensor hA = A.to(kHABANA);
  torch::Tensor hB = B.to(kHABANA);
  torch::Tensor hC = C.to(kHABANA);
  torch::Tensor O = torch::addmm(hA, hB, hC, 1, 1);
  std::string out =
      IrGraphDumpUtil::ToText({GetHbLazyTensor(O).CurrentIrValue().mp_node});
  EXPECT_EQ(
      out.find("IR {\n"
               "  %0 = prim::constant(), value=1\n"
               "  %1 = prim::constant(), value=1\n"
               "  %2 = hpu::input()\n"
               "  %3 = hpu::input()\n"
               "  %4 = hpu::input()\n"
               "  %5 = aten::addmm(%4, %3, %2, %1, %0), ROOT=0\n"
               "}"),
      !std::string::npos);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(O)};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

  auto computed = O.to(torch::kCPU);
  auto expected = torch::addmm(A, B, C, 1, 1);

  EXPECT_EQ(allclose(expected, computed), true);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyLinearKernelTest, BmmTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({4, 2, 3}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({4, 3, 5}, torch::requires_grad(false));
  auto exp = torch::bmm(A, B);
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor result = torch::bmm(hA, hB);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp), true);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyLinearKernelTest, BmmOutTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({4, 2, 3}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({4, 3, 5}, torch::requires_grad(false));
  torch::Tensor out_cpu = torch::randn({4, 2, 5}, torch::requires_grad(false));
  auto exp = torch::bmm(A, B);
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor hOut = out_cpu.to(torch::kHABANA);
  torch::Tensor result = batch_gemm_out_hpu_lazy(hOut, hA, hB);
  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp), true);
  unsetenv("PT_HPU_LAZY_MODE");
}

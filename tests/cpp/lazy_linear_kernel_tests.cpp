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
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyLinearKernelTest, MmMulTest) {
  auto x = torch::randn({2, 3});
  auto y = torch::randn({3, 3});
  auto z = torch::randn({2, 3});
  torch::Tensor hx = x.to(torch::kHABANA);
  torch::Tensor hy = y.to(torch::kHABANA);
  torch::Tensor hz = z.to(torch::kHABANA);

  auto hy_exp = torch::mm(hx, hy);
  auto hz_exp = torch::mul(hy_exp, hz).to(torch::kCPU);

  auto y_cpu = torch::mm(x, y);
  auto z_cout = torch::mul(y_cpu, z);
  EXPECT_EQ(allclose(hz_exp, z_cout), true);
}

TEST_F(LazyLinearKernelTest, AddMmTest) {
  torch::Tensor A = torch::randn({2});
  torch::Tensor B = torch::randn({2, 2});
  torch::Tensor C = torch::randn({2, 2});

  torch::Tensor hA = A.to(kHABANA);
  torch::Tensor hB = B.to(kHABANA);
  torch::Tensor hC = C.to(kHABANA);
  torch::Tensor O = torch::addmm(hA, hB, hC, 1, 1);

  auto computed = O.to(torch::kCPU);
  auto expected = torch::addmm(A, B, C, 1, 1);

  EXPECT_EQ(allclose(expected, computed), true);
}

TEST_F(LazyLinearKernelTest, BmmTest) {
  torch::Tensor A = torch::randn({4, 2, 3}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({4, 3, 5}, torch::requires_grad(false));
  auto exp = torch::bmm(A, B);
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor result = torch::bmm(hA, hB);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp), true);
}

TEST_F(LazyLinearKernelTest, BmmOutTest) {
  torch::Tensor A = torch::randn({4, 2, 3}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({4, 3, 5}, torch::requires_grad(false));
  torch::Tensor out_cpu = torch::randn({4, 2, 5}, torch::requires_grad(false));
  auto exp = torch::bmm_out(out_cpu, A, B);
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor hOut = out_cpu.to(torch::kHABANA);
  torch::Tensor result = batch_gemm_out_hpu_lazy(hOut, hA, hB);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

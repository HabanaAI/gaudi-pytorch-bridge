#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
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

#define rtol 0.001
#define atol 0.001

using namespace habana_lazy;
using namespace at;

class LazyLinearKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyLinearKernelTest, MmMulTest) {
  auto x = torch::randn({2, 3});
  auto y = torch::randn({3, 3});
  auto z = torch::randn({2, 3});
  torch::Tensor hx = x.to(torch::kHPU);
  torch::Tensor hy = y.to(torch::kHPU);
  torch::Tensor hz = z.to(torch::kHPU);

  auto hy_exp = torch::mm(hx, hy);
  auto hz_exp = torch::mul(hy_exp, hz).to(torch::kCPU);

  auto y_cpu = torch::mm(x, y);
  auto z_cout = torch::mul(y_cpu, z);
  EXPECT_TRUE(allclose(hz_exp, z_cout, rtol, atol));
}

TEST_F(LazyLinearKernelTest, AddMmTest) {
  torch::Tensor A = torch::randn({2});
  torch::Tensor B = torch::randn({2, 2});
  torch::Tensor C = torch::randn({2, 2});

  torch::Tensor hA = A.to(kHPU);
  torch::Tensor hB = B.to(kHPU);
  torch::Tensor hC = C.to(kHPU);
  torch::Tensor O = torch::addmm(hA, hB, hC, 1, 1);

  auto computed = O.to(torch::kCPU);
  auto expected = torch::addmm(A, B, C, 1, 1);

  EXPECT_TRUE(allclose(expected, computed, 0.001, 0.001));
}

TEST_F(LazyLinearKernelTest, MatmulTest) {
  auto matmul_test = [](c10::IntArrayRef size1, c10::IntArrayRef size2) {
    auto mat1 = torch::randn(size1).requires_grad_();
    auto mat2 = torch::randn(size2).requires_grad_();
    auto mat1_h = mat1.to(torch::kHPU);
    auto mat2_h = mat2.to(torch::kHPU);

    auto out = torch::matmul(mat1, mat2);
    auto out_h = torch::matmul(mat1_h, mat2_h).to(torch::kCPU);
    EXPECT_TRUE(allclose(out, out_h, 0.01, 0.01));
  };

  matmul_test({10}, {10});
  matmul_test({2, 10}, {10});
  matmul_test({10}, {10, 2});
  matmul_test({2, 10}, {10, 2});
  matmul_test({2, 3, 4}, {4});
  matmul_test({2, 3, 4}, {2, 4, 3});
  matmul_test({12, 20, 24}, {24, 20});
  matmul_test({12, 16, 20, 24}, {12, 16, 24, 20});
  matmul_test({3}, {2, 3, 4});
  matmul_test({3, 4}, {2, 4, 3});
  matmul_test({12, 16, 20, 24}, {16, 24, 20});
  matmul_test({16, 20, 24}, {12, 16, 24, 20});
}

TEST_F(LazyLinearKernelTest, MatmulBwdTest) {
  auto matmulbwd_test = [](c10::IntArrayRef size1, c10::IntArrayRef size2) {
    auto mat1 = torch::randn(size1, torch::requires_grad());
    auto mat2 = torch::randn(size2, torch::requires_grad());
    auto mat1_h = mat1.to(torch::kHPU);
    auto mat2_h = mat2.to(torch::kHPU);
    // retain_grad() as mat1_h and mat2_h are non-leaf tensors
    mat1_h.retain_grad();
    mat2_h.retain_grad();

    auto out = torch::matmul(mat1, mat2);
    auto grad_out = torch::ones_like(out);
    out.backward(grad_out);
    auto grad_mat1 = mat1.grad().clone().detach();
    auto grad_mat2 = mat2.grad().clone().detach();

    auto out_h = torch::matmul(mat1_h, mat2_h);
    auto grad_out_h = grad_out.to(torch::kHPU);
    out_h.backward(grad_out_h);
    auto grad_mat1_h = mat1_h.grad();
    auto grad_mat2_h = mat2_h.grad();

    EXPECT_EQ(
        allclose(grad_mat1, grad_mat1_h.to(torch::kCPU), 0.01, 0.01), true);
    EXPECT_EQ(
        allclose(grad_mat2, grad_mat2_h.to(torch::kCPU), 0.01, 0.01), true);
  };

  matmulbwd_test({2, 3, 4}, {4, 5});
  matmulbwd_test({2, 3, 4}, {2, 4, 5});
  matmulbwd_test({2, 3, 4}, {4});
  matmulbwd_test({2, 2, 3, 4}, {2, 4, 3});
  matmulbwd_test({2, 3}, {3, 4});
  matmulbwd_test({1, 3}, {3, 1});
}

TEST_F(LazyLinearKernelTest, BaddBmmTest1) {
  torch::Tensor A = torch::randn({10, 3, 5});
  torch::Tensor B = torch::randn({10, 3, 4});
  torch::Tensor C = torch::randn({10, 4, 5});
  float beta = 1.0, alpha = 1.0;

  torch::Tensor hA = A.to(kHPU);
  torch::Tensor hB = B.to(kHPU);
  torch::Tensor hC = C.to(kHPU);
  torch::Tensor hComputed = torch::baddbmm(hA, hB, hC, beta, alpha);
  auto expected = torch::baddbmm(A, B, C, beta, alpha);

  auto computed = hComputed.to(torch::kCPU);
  EXPECT_TRUE(allclose(expected, computed, rtol, atol));
}

TEST_F(LazyLinearKernelTest, BaddBmmTest2) {
  torch::Tensor A = torch::randn({10, 3, 5});
  torch::Tensor B = torch::randn({10, 3, 4});
  torch::Tensor C = torch::randn({10, 4, 5});
  float beta = 1.0, alpha = 0.0;

  torch::Tensor hA = A.to(kHPU);
  torch::Tensor hB = B.to(kHPU);
  torch::Tensor hC = C.to(kHPU);
  torch::Tensor hComputed = torch::baddbmm(hA, hB, hC, beta, alpha);
  auto expected = torch::baddbmm(A, B, C, beta, alpha);

  auto computed = hComputed.to(torch::kCPU);
  EXPECT_TRUE(allclose(expected, computed, rtol, atol));
}

TEST_F(LazyLinearKernelTest, BaddBmmTest3) {
  torch::Tensor A = torch::randn({1, 3, 5});
  torch::Tensor B = torch::randn({10, 3, 4});
  torch::Tensor C = torch::randn({10, 4, 5});
  float beta = 0.8, alpha = 0.0;

  torch::Tensor hA = A.to(kHPU);
  torch::Tensor hB = B.to(kHPU);
  torch::Tensor hC = C.to(kHPU);
  torch::Tensor hComputed = torch::baddbmm(hA, hB, hC, beta, alpha);
  auto expected = torch::baddbmm(A, B, C, beta, alpha);

  auto computed = hComputed.to(torch::kCPU);
  EXPECT_TRUE(allclose(expected, computed, rtol, atol));
}

TEST_F(LazyLinearKernelTest, BaddBmmTest4) {
  torch::Tensor A = torch::randn({3, 5});
  torch::Tensor B = torch::randn({10, 3, 4});
  torch::Tensor C = torch::randn({10, 4, 5});
  float beta = 0.8, alpha = 0.2;

  torch::Tensor hA = A.to(kHPU);
  torch::Tensor hB = B.to(kHPU);
  torch::Tensor hC = C.to(kHPU);
  torch::Tensor hComputed = torch::baddbmm(hA, hB, hC, beta, alpha);
  auto expected = torch::baddbmm(A, B, C, beta, alpha);

  auto computed = hComputed.to(torch::kCPU);
  EXPECT_TRUE(allclose(expected, computed, rtol, atol));
}

TEST_F(LazyLinearKernelTest, BaddBmmOutTest1) {
  torch::Tensor A = torch::randn({3, 5});
  torch::Tensor B = torch::randn({10, 3, 4});
  torch::Tensor C = torch::randn({10, 4, 5});
  torch::Tensor out_cpu = torch::randn({10, 3, 5});
  float beta = 1.0, alpha = 1.0;

  torch::Tensor hA = A.to(kHPU);
  torch::Tensor hB = B.to(kHPU);
  torch::Tensor hC = C.to(kHPU);
  torch::Tensor hOut = out_cpu.to(kHPU);
  torch::Tensor hComputed = torch::baddbmm_out(hOut, hA, hB, hC, beta, alpha);
  auto expected = torch::baddbmm_out(out_cpu, A, B, C, beta, alpha);

  auto computed = hComputed.to(torch::kCPU);
  EXPECT_TRUE(allclose(expected, computed, rtol, atol));
}

TEST_F(LazyLinearKernelTest, BaddBmmOutTest2) {
  torch::Tensor A = torch::randn({10, 3, 5});
  torch::Tensor B = torch::randn({10, 3, 4});
  torch::Tensor C = torch::randn({10, 4, 5});
  torch::Tensor out_cpu = torch::randn({10, 3, 5});
  float beta = 1.2, alpha = 0.0;

  torch::Tensor hA = A.to(kHPU);
  torch::Tensor hB = B.to(kHPU);
  torch::Tensor hC = C.to(kHPU);
  torch::Tensor hOut = out_cpu.to(kHPU);
  torch::baddbmm_out(hOut, hA, hB, hC, beta, alpha);
  torch::baddbmm_out(out_cpu, A, B, C, beta, alpha);

  auto computed = hOut.to(torch::kCPU);
  EXPECT_TRUE(allclose(out_cpu, computed, rtol, atol));
}

TEST_F(LazyLinearKernelTest, BaddBmmInplaceTest1) {
  torch::Tensor A = torch::randn({10, 3, 5});
  torch::Tensor B = torch::randn({10, 3, 4});
  torch::Tensor C = torch::randn({10, 4, 5});
  float beta = 0.6, alpha = 0.3;

  torch::Tensor hA = A.to(kHPU);
  torch::Tensor hB = B.to(kHPU);
  torch::Tensor hC = C.to(kHPU);
  hA.baddbmm_(hB, hC, beta, alpha);
  A.baddbmm_(B, C, beta, alpha);

  auto computed = hA.to(torch::kCPU);
  EXPECT_TRUE(allclose(A, computed, rtol, atol));
}

TEST_F(LazyLinearKernelTest, BaddBmmInplaceTest2) {
  torch::Tensor A = torch::randn({10, 3, 5});
  torch::Tensor B = torch::randn({10, 3, 4});
  torch::Tensor C = torch::randn({10, 4, 5});
  float beta = 0.6, alpha = 0.0;

  torch::Tensor hA = A.to(kHPU);
  torch::Tensor hB = B.to(kHPU);
  torch::Tensor hC = C.to(kHPU);
  hA.baddbmm_(hB, hC, beta, alpha);
  A.baddbmm_(B, C, beta, alpha);

  auto computed = hA.to(torch::kCPU);
  EXPECT_TRUE(allclose(A, computed, rtol, atol));
}

TEST_F(LazyLinearKernelTest, BaddBmmInplaceTest3) {
  torch::Tensor A = torch::randn({10, 3, 5});
  torch::Tensor B = torch::randn({10, 3, 4});
  torch::Tensor C = torch::randn({10, 4, 5});
  float beta = 0.0, alpha = 0.8;

  torch::Tensor hA = A.to(kHPU);
  torch::Tensor hB = B.to(kHPU);
  torch::Tensor hC = C.to(kHPU);
  hA.baddbmm_(hB, hC, beta, alpha);
  A.baddbmm_(B, C, beta, alpha);

  auto computed = hA.to(torch::kCPU);
  EXPECT_TRUE(allclose(A, computed, rtol, atol));
}

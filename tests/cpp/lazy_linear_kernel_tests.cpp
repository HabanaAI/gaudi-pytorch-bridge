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

TEST_F(LazyLinearKernelTest, MatmulTest) {
  auto matmul_test = [](c10::IntArrayRef size1, c10::IntArrayRef size2) {
    auto mat1 = torch::randn(size1).requires_grad_();
    auto mat2 = torch::randn(size2).requires_grad_();
    auto mat1_h = mat1.to(torch::kHABANA);
    auto mat2_h = mat2.to(torch::kHABANA);

    auto out = torch::matmul(mat1, mat2);
    auto out_h = torch::matmul(mat1_h, mat2_h).to(torch::kCPU);

    EXPECT_EQ(allclose(out, out_h, 0.01, 0.01), true);
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

/*
 * Commenting out cpp test for matmul backward for now, due to an error in Test
 * Case code. Added a python unit test at
 * pytorch-integration/tests/test_lazy_matmul.py
 */
/*
TEST_F(LazyLinearKernelTest, MatmulBwdTest) {
  auto matmulbwd_test = [](c10::IntArrayRef size1, c10::IntArrayRef size2) {
    auto mat1 = torch::randn(size1, torch::requires_grad());
    auto mat2 = torch::randn(size2, torch::requires_grad());
    auto mat1_h = mat1.to(torch::kHABANA);
    auto mat2_h = mat2.to(torch::kHABANA);

    auto out = torch::matmul(mat1, mat2);

    auto grad_out = torch::ones_like(out);
    //auto grad_out_h = grad_out.to(torch::kHABANA);
    out.backward(grad_out);
    auto grad_mat1 = mat1.grad();
    auto grad_mat2 = mat2.grad();

    auto out_h = torch::matmul(mat1_h, mat2_h);
    auto grad_out_h = grad_out.to(torch::kHABANA);
    out_h.backward(grad_out_h);
    auto grad_mat1_h = mat1_h.grad();
    auto grad_mat2_h = mat2_h.grad();
    std::cout << "$$ grad_mat1 - " << grad_mat1 << std::endl;
    std::cout << mat1_h.to(torch::kCPU).sizes().vec() << std::endl;

    // torch::Tensor grad_mat1_h, grad_mat2_h;
    // std::tie(grad_mat1_h, grad_mat2_h) =
    //    hpu_wrap::matmul_backward(grad_out_h, mat1_h, mat2_h);

    HbLazyTensor::StepMarker({});

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
}
*/

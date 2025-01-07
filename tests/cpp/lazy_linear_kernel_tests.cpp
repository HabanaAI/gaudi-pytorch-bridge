/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
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
#include "utils/device_type_util.h"

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
  matmul_test({10, 8, 16}, {1, 16, 12});
  matmul_test({2, 10, 8, 16}, {2, 1, 16, 12});

  // testing all broadcast scenarios
  // all combinations of batch dim sizes=[1..2] across all different rank
  // configurations
  auto generator = [](int dim, int gen, c10::IntArrayRef last_dims) {
    auto mask = [](int i, int idx) { return (i & 1 << idx) ? 2 : 1; };

    std::vector<int64_t> ret(dim);
    if (dim == 1) {
      ret[0] = 4;
    } else {
      ret[dim - 1] = last_dims[1];
      ret[dim - 2] = last_dims[0];
      for (int i = 0; i < dim - 2; i++) {
        ret[i] = mask(gen, i);
      }
    }
    return ret;
  };

  // iterate over all combinations of N-D x M-D; N,M in range [1..5]
  for (int N = 1; N <= 5; N++) {
    for (int M = 1; M <= 5; M++) {
      // now iterate over all cases for each N and M
      for (int gen1 = 0; gen1 < 1 << std::max(0, N - 2); gen1++) {
        for (int gen2 = 0; gen2 < 1 << std::max(0, M - 2); gen2++) {
          // perform the test for each case
          matmul_test(generator(N, gen1, {3, 4}), generator(M, gen2, {4, 5}));
        }
      }
    }
  }
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

  // testing all broadcast scenarios
  // all combinations of batch dim sizes=[1..2] across all different rank
  // configurations
  auto generator = [](int dim, int gen, c10::IntArrayRef last_dims) {
    auto mask = [](int i, int idx) { return (i & 1 << idx) ? 2 : 1; };

    std::vector<int64_t> ret(dim);
    if (dim == 1) {
      ret[0] = 4;
    } else {
      ret[dim - 1] = last_dims[1];
      ret[dim - 2] = last_dims[0];
      for (int i = 0; i < dim - 2; i++) {
        ret[i] = mask(gen, i);
      }
    }
    return ret;
  };

  auto should_skip_case_on_gaudi = [](int N, int M, int gen1, int gen2) {
    if (N == 3 && M == 5 && gen1 == 1 && (gen2 == 1 || gen2 == 2 || gen2 == 3))
      return true;
    if (N == 4 && M == 5 && (gen1 == 1 || gen1 == 2 || gen1 == 3) && gen2 == 1)
      return true;
    if (N == 4 && M == 5 && gen1 == 3 && (gen2 == 3 || gen2 == 5))
      return true;
    if (N == 5 && M == 3 && (gen1 == 1 || gen1 == 2 || gen1 == 3) && gen2 == 1)
      return true;
    if (N == 5 && M == 4 && gen1 == 1 && (gen2 == 1 || gen2 == 2 || gen2 == 3))
      return true;
    if (N == 5 && M == 4 && (gen1 == 3 || gen1 == 5) && gen2 == 3)
      return true;

    return false;
  };

  // iterate over all combinations of N-D x M-D; N,M in range [1..5]
  for (int N = 1; N <= 5; N++) {
    for (int M = 1; M <= 5; M++) {
      // now iterate over all cases for each N and M
      for (int gen1 = 0; gen1 < 1 << std::max(0, N - 2); gen1++) {
        for (int gen2 = 0; gen2 < 1 << std::max(0, M - 2); gen2++) {
          if (isGaudi() && should_skip_case_on_gaudi(N, M, gen1, gen2))
            continue;
          // perform the test for each case
          matmulbwd_test(
              generator(N, gen1, {3, 4}), generator(M, gen2, {4, 5}));
        }
      }
    }
  }
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

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
#include "generated/lazy/wrap_kernels_declarations.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
class MatmulBwdTest : public habana_lazy_test::LazyTest {
 public:
  void matmulBwdTest(at::IntArrayRef inputAShape, at::IntArrayRef inputBShape) {
    auto in_a = torch::randn(inputAShape, torch::requires_grad());
    auto hin_a = in_a.to(torch::kHPU);
    auto in_b = torch::randn(inputBShape, torch::requires_grad());
    auto hin_b = in_b.to(torch::kHPU);
    auto exp = torch::matmul(in_a, in_b);
    auto exp_hpu = habana_lazy::matmul_hpu_lazy(hin_a, hin_b);

    auto grad_out = exp.detach();
    auto hgrad_out = grad_out.detach().to(torch::kHPU);

    exp.backward(grad_out);

    auto grad_in_a = in_a.grad();
    auto grad_in_b = in_b.grad();
    at::Tensor hgrad_in_a, hgrad_in_b;
    std::tie(hgrad_in_a, hgrad_in_b) =
        habana_lazy::matmul_backward_hpu_lazy(hgrad_out, hin_a, hin_b);
    auto hgrad_in_a_cpu = hgrad_in_a.to(torch::kCPU);
    auto hgrad_in_b_cpu = hgrad_in_b.to(torch::kCPU);
    EXPECT_EQ(allclose(grad_in_a, hgrad_in_a_cpu, 0.001, 0.001), true);
    EXPECT_EQ(allclose(grad_in_b, hgrad_in_b_cpu, 0.001, 0.001), true);
  }
};

TEST_F(MatmulBwdTest, MatmulTest1d1d) {
  matmulBwdTest({2}, {2});
}
TEST_F(MatmulBwdTest, MatmulTest1d2d) {
  matmulBwdTest({2}, {2, 4});
}
TEST_F(MatmulBwdTest, MatmulTest1d3d) {
  matmulBwdTest({2}, {5, 2, 4});
}
TEST_F(MatmulBwdTest, MatmulTest1d4d) {
  matmulBwdTest({2}, {3, 5, 2, 4});
}
TEST_F(MatmulBwdTest, MatmulTest1d5d) {
  matmulBwdTest({5}, {6, 3, 5, 5, 4});
}
TEST_F(MatmulBwdTest, MatmulTest2d1d) {
  matmulBwdTest({4, 2}, {2});
}
TEST_F(MatmulBwdTest, MatmulTest2d2d) {
  matmulBwdTest({3, 4}, {4, 5});
}
TEST_F(MatmulBwdTest, MatmulTest2d3d) {
  matmulBwdTest({15, 4}, {5, 4, 3});
}
TEST_F(MatmulBwdTest, MatmulTest2d4d) {
  matmulBwdTest({4, 2}, {5, 5, 2, 10});
}
TEST_F(MatmulBwdTest, MatmulTest2d5d) {
  matmulBwdTest({4, 2}, {5, 5, 5, 2, 3});
}
TEST_F(MatmulBwdTest, MatmulTest3d1d) {
  matmulBwdTest({5, 4, 3}, {3});
}
TEST_F(MatmulBwdTest, MatmulTest3d2d) {
  matmulBwdTest({5, 4, 3}, {3, 4});
}
TEST_F(MatmulBwdTest, MatmulTest3d3d) {
  matmulBwdTest({5, 4, 2}, {5, 2, 4});
}
TEST_F(MatmulBwdTest, MatmulTest4d1d) {
  matmulBwdTest({4, 5, 4, 6}, {6});
}
TEST_F(MatmulBwdTest, MatmulTest4d2d) {
  matmulBwdTest({4, 5, 4, 6}, {6, 4});
}
TEST_F(MatmulBwdTest, MatmulTest4d4d) {
  matmulBwdTest({3, 2, 3, 4}, {3, 2, 4, 5});
}
TEST_F(MatmulBwdTest, MatmulTest5d1d) {
  matmulBwdTest({1, 4, 5, 4, 2}, {2});
}
TEST_F(MatmulBwdTest, MatmulTest5d2d) {
  matmulBwdTest({1, 4, 5, 4, 2}, {2, 4});
}
TEST_F(MatmulBwdTest, MatmulTest5d3d) {
  matmulBwdTest({1, 4, 5, 4, 2}, {5, 2, 4});
}
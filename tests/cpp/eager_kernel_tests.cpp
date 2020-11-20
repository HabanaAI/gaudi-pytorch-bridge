#include <gtest/gtest.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/linear_kernels.h"

#include "habana_kernels/eager_kernels_declarations.h"

TEST(EagerKernelTest, ReluTest) {
  torch::Tensor tensor = torch::randn({2, 3});
  torch::Tensor tHabana = tensor.to(torch::kHABANA);
  auto outHabana = torch::relu(tHabana);
  auto out = torch::relu(tensor);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST(EagerKernelTest, AddTest) {
  torch::Tensor tensor = torch::randn({2, 3});
  torch::Tensor tHabana = tensor.to(torch::kHABANA);
  auto outHabana = torch::add(tHabana, 4.0);
  auto out = torch::add(tensor, 4.0);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST(EagerKernelTest, MatMulTest) {
  auto matmul_test = [](c10::IntArrayRef size1, c10::IntArrayRef size2) {
    torch::Tensor tensor1 = torch::randn(size1);
    torch::Tensor tensor2 = torch::randn(size2); // torch::randn({2, 2}); d2 =1,2 tested and passing
    torch::Tensor ht1 = tensor1.to(torch::kHABANA);
    torch::Tensor ht2 = tensor2.to(torch::kHABANA);
    auto outHabana = matmul_hpu(ht1, ht2);
    auto out = torch::matmul(tensor1, tensor2);
    bool equal = out.allclose(outHabana.to(torch::kCPU), 0.001, 0.001);
    EXPECT_EQ(equal, true);
  };

  matmul_test({2,2,2}, {2});
  matmul_test({2,2,2}, {2,2,2});
  matmul_test({12, 384, 1024}, {1024, 4096});
  matmul_test({12, 384, 768}, {768, 768});
  matmul_test({12, 384, 1024}, {1024, 1024});
  matmul_test({12, 16, 384, 64}, {12, 16, 64, 384});

}

TEST(EagerKernelTest, MatmulBackwardTest) {
  torch::manual_seed(0);
  auto matmul_test = [](c10::IntArrayRef size1, c10::IntArrayRef size2) {
    auto mat1 = torch::randn(size1);
    auto mat2 = torch::randn(size2);
    auto mat1_h = mat1.to(torch::kHABANA);
    auto mat2_h = mat2.to(torch::kHABANA);

    mat1.set_requires_grad(true);
    mat2.set_requires_grad(true);
    auto out = torch::matmul(mat1, mat2);

    auto grad_out = torch::ones_like(out);
    auto grad_out_h = grad_out.to(torch::kHABANA);
    out.backward(grad_out);
    auto grad_mat1 = mat1.grad();
    auto grad_mat2 = mat2.grad();

    torch::Tensor grad_mat1_h, grad_mat2_h;
    std::tie(grad_mat1_h, grad_mat2_h) =
        matmul_backward_hpu(grad_out_h, mat1_h, mat2_h);
    bool equal1 = grad_mat1.allclose(grad_mat1_h.to(torch::kCPU), 0.01, 0.01);
    EXPECT_EQ(equal1, true);
    bool equal2 = grad_mat2.allclose(grad_mat2_h.to(torch::kCPU), 0.01, 0.01);
    EXPECT_EQ(equal2, true);
  };
  matmul_test({2, 3, 4}, {4, 5});
  matmul_test({2, 3, 4}, {2, 4, 5});
  matmul_test({2, 3, 4}, {4});
  matmul_test({2, 2, 3, 4}, {2, 4, 3});
}
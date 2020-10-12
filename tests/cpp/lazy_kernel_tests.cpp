#include <gtest/gtest.h>
#include <torch/torch.h>
#include <stdexcept>

TEST(LazyKernelTest, LazyDoATest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor C = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor hC = C.to(torch::kHABANA);
  torch::Tensor I  = torch::add(hA, hB);
  torch::Tensor out = torch::add(hC, I);
  unsetenv("PT_HPU_LAZY_MODE");
  //bool equal = out.allclose(out.to(torch::kCPU), 0, 0);
  EXPECT_EQ(out.dim(), 2);
}

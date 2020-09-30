#include <gtest/gtest.h>
#include <torch/torch.h>
#include <stdexcept>

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

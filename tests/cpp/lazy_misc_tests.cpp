#include <gtest/gtest.h>
#include <torch/torch.h>
#include "tests/cpp/habana_lazy_test_infra.h"

using namespace habana_lazy;
using namespace at;

class LazyMiscTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyMiscTest, CatchExceptionTest) {
  auto x = torch::randn({2, 3});
  auto y1 = torch::randn({4, 3});

  torch::Tensor hx = x.to(torch::kHPU);
  torch::Tensor hy1 = y1.to(torch::kHPU);

  try {
    auto z = torch::mm(hx, hy1).to(torch::kCPU);
  } catch (...) {
    auto y2 = torch::randn({3, 3});
    torch::Tensor hy2 = y2.to(torch::kHPU);
    auto z = torch::mm(hx, hy2).to(torch::kCPU);
    auto z_cpu = torch::mm(x, y2);
    EXPECT_EQ(allclose(z, z_cpu, 0.001, 0.001), true);
    return;
  }
  EXPECT_EQ(false, true);
}

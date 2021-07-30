#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <cstdlib>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;
using namespace at;

class LazyWhereKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyWhereKernelTest, WhereTest) {
  torch::Tensor x = torch::randn({2, 3});
  torch::Tensor y = torch::randn({2, 3});
  auto out = torch::_s_where(x > 0, x, y);

  auto hx = x.to(torch::kHPU);
  auto hy = y.to(torch::kHPU);
  auto outHabana = torch::_s_where(hx > 0, hx, hy);

  auto result = outHabana.to(torch::kCPU);

  bool equal = out.allclose(result, 0.001, 0.001);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyWhereKernelTest, WhereBroadcastTest) {
  torch::Tensor cond = torch::randint(0, 2, {2, 3});
  torch::Tensor condBool = cond > 0;
  torch::Tensor x = torch::randn({2, 3});
  torch::Tensor y = torch::randn({1});

  auto out = torch::_s_where(condBool, x, y);

  auto hcond = condBool.to(torch::kHPU);
  auto hx = x.to(torch::kHPU);
  auto hy = y.to(torch::kHPU);
  auto outHabana = torch::_s_where(hcond, hx, hy);

  auto result = outHabana.to(torch::kCPU);

  bool equal = out.allclose(result, 0.001, 0.001);
  EXPECT_EQ(equal, true);
}

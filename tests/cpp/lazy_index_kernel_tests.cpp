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

class LazyIndexKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyIndexKernelTest, IndexSelectTest) {
  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHABANA);
  int64_t dim = 1;
  auto index = torch::tensor({0, 1}, torch::dtype(torch::kInt64));
  auto h_index = index.to(torch::kHABANA);

  Tensor h_out = torch::index_select(h_a, dim, h_index);

  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::index_select(a, dim, index);

  EXPECT_EQ(allclose(h_cout, cout), true);
}

TEST_F(LazyIndexKernelTest, IndexAddInplaceTest) {
  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHABANA);
  int64_t dim = 1;
  auto index = torch::tensor({0, 1}, torch::dtype(torch::kInt64));
  auto h_index = index.to(torch::kHABANA);
  auto source = torch::randn({8, 2, 28, 28}, torch::requires_grad(false));
  auto h_source = source.to(torch::kHABANA);

  h_a.index_add_(dim, h_index, h_source);
  auto h_temp = torch::zeros({8, 3, 28, 28}).to(torch::kHABANA);
  ;
  auto out = torch::add(h_a, h_temp);

  auto h_cout = out.to(torch::kCPU);

  a.index_add_(dim, index, source);

  EXPECT_EQ(allclose(h_cout, a), true);
}

TEST_F(LazyIndexKernelTest, ScatterValueInplaceTest) {
  torch::Tensor a = torch::randn({5, 7}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHABANA);
  int64_t dim = 0;
  auto index = torch::randint(0, 5, {5, 7}, torch::dtype(torch::kInt64));
  auto h_index = index.to(torch::kHABANA);
  auto value = 2;

  h_a.scatter_(dim, h_index, value);
  auto h_cout = h_a.to(torch::kCPU);
  a.scatter_(dim, index, value);

  EXPECT_EQ(allclose(h_cout, a), true);
}

TEST_F(LazyIndexKernelTest, ScatterValueTest) {
  torch::Tensor a = torch::randn({5, 7}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHABANA);
  int64_t dim = 0;
  auto index = torch::randint(0, 5, {5, 7}, torch::dtype(torch::kInt64));
  auto h_index = index.to(torch::kHABANA);
  auto value = 2;

  torch::Tensor hOut = torch::scatter(h_a, dim, h_index, value);
  auto h_cout = hOut.to(torch::kCPU);
  torch::Tensor out = torch::scatter(a, dim, index, value);

  EXPECT_EQ(allclose(h_cout, out), true);
}

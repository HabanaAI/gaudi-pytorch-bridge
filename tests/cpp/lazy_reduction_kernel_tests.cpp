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

class LazyReductionKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyReductionKernelTest, SumTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = torch::sum(hA);
  torch::Tensor Out = torch::sum(A);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.001, 0.001), true);
}

TEST_F(LazyReductionKernelTest, MeanDim) {
  torch::Tensor A = torch::randn({53, 13}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = at::mean(hA, {0});
  torch::Tensor Out = at::mean(A, {0});

  EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out));
}
TEST_F(LazyReductionKernelTest, SumDimIntTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = torch::sum(hA, 1);
  torch::Tensor Out = torch::sum(A, 1);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyReductionKernelTest, ProdTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = torch::prod(hA);
  torch::Tensor Out = torch::prod(A);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.001, 0.001), true);
}

TEST_F(LazyReductionKernelTest, ProdDimIntTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = torch::prod(hA, 1);
  torch::Tensor Out = torch::prod(A, 1);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyReductionKernelTest, ArgMaxTest) {
  torch::Tensor A = torch::randn({2, 2, 3, 4}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = torch::argmax(hA, 2, true);
  torch::Tensor Out = torch::argmax(A, 2, true);
  auto cOut = Out.to(torch::dtype(torch::kInt));
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), cOut), true);
}

TEST_F(LazyReductionKernelTest, AllTensorTest) {
  exec::OptPassCfg::GetInstance()->enable_subgraph_rewrite = true;
  const std::vector<int64_t> dimensions{5, 3, 4};

  torch::Tensor A = (torch::randn(dimensions) > 0.5);

  auto expected = torch::all(A);
  auto hA = A.to(torch::kHABANA);
  auto result = torch::all(hA);
  torch::Tensor habanaGenerated = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(expected.to(torch::kInt8), habanaGenerated.to(torch::kInt8)),
      true);
  exec::OptPassCfg::GetInstance()->enable_subgraph_rewrite = false;
}

TEST_F(LazyReductionKernelTest, AllDimTensorTest) {
  exec::OptPassCfg::GetInstance()->enable_subgraph_rewrite = true;
  const std::vector<int64_t> dimensions{5, 3, 4};

  torch::Tensor A = (torch::randn(dimensions) > 0.5);
  int64_t dim = 1;
  bool keepdim = false;
  auto expected = torch::all(A, dim, keepdim);
  auto hA = A.to(torch::kHABANA);
  auto result = torch::all(hA, dim, keepdim);
  torch::Tensor habanaGenerated = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(expected.to(torch::kInt8), habanaGenerated.to(torch::kInt8)),
      true);
  exec::OptPassCfg::GetInstance()->enable_subgraph_rewrite = false;
}

TEST_F(LazyReductionKernelTest, MaxDimTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut, hIndex, Out, Index;
  std::tie(hOut, hIndex) = torch::max(hA, 1);
  std::tie(Out, Index) = torch::max(A, 1);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
  EXPECT_EQ(allclose(hIndex.to(torch::kCPU).to(torch::kLong), Index), true);
}

TEST_F(LazyReductionKernelTest, MaxTest) {
  torch::Tensor A = torch::randn({2, 3, 4}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  auto hOut = torch::max(hA);
  auto Out = torch::max(A);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

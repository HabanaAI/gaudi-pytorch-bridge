#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
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
using namespace at;

class LazyReductionKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyReductionKernelTest, SumTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::sum(hA);
  torch::Tensor Out = torch::sum(A);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.001, 0.001), true);
}

TEST_F(LazyReductionKernelTest, MeanDim) {
  torch::Tensor A = torch::randn({53, 13}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = at::mean(hA, {0});
  torch::Tensor Out = at::mean(A, {0});

  EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out, 0.001, 0.001));
}
TEST_F(LazyReductionKernelTest, SumDimIntTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::sum(hA, 1);
  torch::Tensor Out = torch::sum(A, 1);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.001, 0.001), true);
}

TEST_F(LazyReductionKernelTest, SumDimIntOut) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);

  torch::Tensor hOut = at::empty_like(hA);
  torch::Tensor Out = at::empty_like(A);

  torch::Tensor out_cpu = torch::sum_outf(A, {0}, false, c10::nullopt, Out);
  torch::Tensor out_hpu = torch::sum_outf(hA, {0}, false, c10::nullopt, hOut);
  auto hOut_cpu = out_hpu.to(torch::kCPU);
  EXPECT_EQ(
      allclose(hOut_cpu, out_cpu, COMMON_ATOL_FLOAT, COMMON_RTOL_FLOAT), true);
}

TEST_F(LazyReductionKernelTest, AllDimOut) {
  torch::Tensor A = torch::randint(-10, 10, {3, 2}) > 0;
  torch::Tensor hA = A.to(torch::kHPU);

  torch::Tensor hOut = at::empty_like(hA);
  torch::Tensor Out = at::empty_like(A);

  torch::Tensor out_cpu = torch::all_outf(A, 0, false, Out);
  torch::Tensor out_hpu = torch::all_outf(hA, 0, false, hOut);
  auto hOut_cpu = out_hpu.to(torch::kCPU);
  EXPECT_EQ(allclose(hOut_cpu.to(torch::kI8), out_cpu.to(torch::kI8)), true);
}

TEST_F(LazyReductionKernelTest, ProdTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::prod(hA);
  torch::Tensor Out = torch::prod(A);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.001, 0.001), true);
}

TEST_F(LazyReductionKernelTest, AnyTest) {
  torch::Tensor A = torch::randint(-10, 10, {3, 2}) > 0;
  torch::Tensor hA = A.to(torch::kHPU);

  torch::Tensor out = torch::any(hA);
  torch::Tensor out_cpu = torch::any(A);
  torch::Tensor out_h = out.to(torch::kCPU);

  EXPECT_EQ(allclose(out_h.to(torch::kI8), out_cpu.to(torch::kI8)), true);
}

TEST_F(LazyReductionKernelTest, AnyDimTest) {
  torch::Tensor A = torch::randint(-10, 10, {3, 2}) > 0;
  torch::Tensor hA = A.to(torch::kHPU);

  torch::Tensor out = torch::any(hA, 1, true);
  torch::Tensor out_cpu = torch::any(A, 1, true);
  torch::Tensor out_h = out.to(torch::kCPU);

  EXPECT_EQ(allclose(out_h.to(torch::kI8), out_cpu.to(torch::kI8)), true);
}

TEST_F(LazyReductionKernelTest, AnyDimOutTest) {
  torch::Tensor A = torch::randint(-10, 10, {3, 2}) > 0;
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor A_out = torch::randn({3, 2}) > 0;
  torch::Tensor hA_out = A_out.to(torch::kHPU);

  torch::Tensor out = torch::any_outf(hA, 1, true, hA_out);
  torch::Tensor out_cpu = torch::any_outf(A, 1, true, A_out);
  torch::Tensor out_h = out.to(torch::kCPU);

  EXPECT_TRUE(out_h.equal(out_cpu));
}

TEST_F(LazyReductionKernelTest, ProdDimIntTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::prod(hA, 1);
  torch::Tensor Out = torch::prod(A, 1);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out, 0.001, 0.001), true);
}

TEST_F(LazyReductionKernelTest, ArgMaxTest) {
  torch::Tensor A = torch::randn({2, 2, 3, 4}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::argmax(hA, 2, true);
  torch::Tensor Out = torch::argmax(A, 2, true);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyReductionKernelTest, ArgMaxTestNe1) {
  int dimReduction = -1;
  torch::Tensor A = torch::randn({2, 2, 3, 4}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);

  torch::Tensor Out = torch::argmax(A, dimReduction, true);
  torch::Tensor hOut = torch::argmax(hA, dimReduction, true);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyReductionKernelTest, AllTensorTest) {
  const std::vector<int64_t> dimensions{5, 3, 4};

  torch::Tensor A = (torch::randn(dimensions) > 0.5);

  auto expected = torch::all(A);
  auto hA = A.to(torch::kHPU);
  auto result = torch::all(hA);
  torch::Tensor habanaGenerated = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(expected.to(torch::kInt8), habanaGenerated.to(torch::kInt8)),
      true);
}

TEST_F(LazyReductionKernelTest, AllDimTensorTest) {
  const std::vector<int64_t> dimensions{5, 3, 4};

  torch::Tensor A = (torch::randn(dimensions) > 0.5);
  int64_t dim = 1;
  bool keepdim = false;
  auto expected = torch::all(A, dim, keepdim);
  auto hA = A.to(torch::kHPU);
  auto result = torch::all(hA, dim, keepdim);
  torch::Tensor habanaGenerated = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(expected.to(torch::kInt8), habanaGenerated.to(torch::kInt8)),
      true);
}

TEST_F(LazyReductionKernelTest, MaxDimTest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut, hIndex, Out, Index;
  std::tie(hOut, hIndex) = torch::max(hA, 1);
  std::tie(Out, Index) = torch::max(A, 1);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
  EXPECT_EQ(allclose(hIndex.to(torch::kCPU).to(torch::kLong), Index), true);
}

TEST_F(LazyReductionKernelTest, MaxTest) {
  torch::Tensor A = torch::randn({2, 3, 4}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  auto hOut = torch::max(hA);
  auto Out = torch::max(A);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyReductionKernelTest, MinTest) {
  torch::Tensor A = torch::randn(
      {2, 3, 4, 2}, torch::requires_grad(false).dtype(torch::kDouble));
  torch::Tensor hA = A.to(torch::kHPU);
  auto hOut = torch::min(hA);
  auto Out = torch::min(A);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyReductionKernelTest, DISABLED_Mean) {
  torch::Tensor A = torch::randn({53, 13}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = torch::mean(hA);
  torch::Tensor Out = torch::mean(A);

  EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out));
}

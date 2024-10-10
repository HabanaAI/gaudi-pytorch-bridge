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

using namespace habana_lazy;
using namespace at;

class LazyCompareKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyCompareKernelTest, EqScalarTest) {
  torch::Tensor A = torch::rand({2, 2}, torch::requires_grad(false));
  float compVal = 1.1f;
  auto out_cpu = torch::eq(A, compVal);

  auto hA = A.to(torch::kHPU);
  auto result = torch::eq(hA, compVal);
  torch::Tensor out_hpu = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(out_cpu.to(torch::kFloat), out_hpu.to(torch::kFloat)), true);
}

TEST_F(LazyCompareKernelTest, EqTensorTest) {
  const std::vector<int64_t> dimentions{5, 3, 4};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);

  auto expected = torch::eq(A, B);
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  auto result = torch::eq(hA, hB);
  torch::Tensor habanaGenerated = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(expected.to(torch::kInt8), habanaGenerated.to(torch::kInt8)),
      true);
}

TEST_F(LazyCompareKernelTest, LtScalarTest) {
  torch::Tensor A = torch::rand({2, 2}, torch::requires_grad(false));
  float compVal = 1.1f;
  auto out_cpu = torch::lt(A, compVal);

  auto hA = A.to(torch::kHPU);
  auto result = torch::lt(hA, compVal);
  torch::Tensor out_hpu = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(out_cpu.to(torch::kFloat), out_hpu.to(torch::kFloat)), true);
}

TEST_F(LazyCompareKernelTest, LtTensorTest) {
  const std::vector<int64_t> dimentions{5, 3, 4};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);

  auto expected = torch::lt(A, B);
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  auto result = torch::lt(hA, hB);
  torch::Tensor habanaGenerated = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(expected.to(torch::kInt8), habanaGenerated.to(torch::kInt8)),
      true);
}

TEST_F(LazyCompareKernelTest, GeScalarTest) {
  torch::Tensor A = torch::rand({2, 2}, torch::requires_grad(false));
  float compVal = 1.1f;
  auto out_cpu = torch::ge(A, compVal);

  auto hA = A.to(torch::kHPU);
  auto result = torch::ge(hA, compVal);
  torch::Tensor out_hpu = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(out_cpu.to(torch::kFloat), out_hpu.to(torch::kFloat)), true);
}

TEST_F(LazyCompareKernelTest, GeTensorTest) {
  const std::vector<int64_t> dimentions{5, 3, 4};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);

  auto expected = torch::ge(A, B);
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  auto result = torch::ge(hA, hB);
  torch::Tensor habanaGenerated = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(expected.to(torch::kInt8), habanaGenerated.to(torch::kInt8)),
      true);
}

TEST_F(LazyCompareKernelTest, LeScalarTest) {
  torch::Tensor A = torch::rand({2, 2}, torch::requires_grad(false));
  float compVal = 1.1f;
  auto out_cpu = torch::le(A, compVal);

  auto hA = A.to(torch::kHPU);
  auto result = torch::le(hA, compVal);
  torch::Tensor out_hpu = result.to(torch::kCPU);

  EXPECT_TRUE(out_cpu.equal(out_hpu));
}

TEST_F(LazyCompareKernelTest, LeTensorTest) {
  const std::vector<int64_t> dimentions{5, 3, 4};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::tensor(1.0);

  auto expected = torch::le(A, B);
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  auto result = torch::le(hA, hB);
  torch::Tensor habanaGenerated = result.to(torch::kCPU);

  EXPECT_TRUE(expected.equal(habanaGenerated));
}

TEST_F(LazyCompareKernelTest, NeScalarTest) {
  torch::Tensor A = torch::rand({2, 2}, torch::requires_grad(false));
  float compVal = 1.1f;
  auto out_cpu = torch::ne(A, compVal);

  auto hA = A.to(torch::kHPU);
  auto result = torch::ne(hA, compVal);
  torch::Tensor out_hpu = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(out_cpu.to(torch::kFloat), out_hpu.to(torch::kFloat)), true);
}

TEST_F(LazyCompareKernelTest, NeTensorTest) {
  const std::vector<int64_t> dimensions{5, 3, 4};

  torch::Tensor A = torch::randn(dimensions);
  torch::Tensor B = torch::randn(dimensions);

  auto expected = torch::ne(A, B);
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  auto result = torch::ne(hA, hB);
  torch::Tensor habanaGenerated = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(expected.to(torch::kInt8), habanaGenerated.to(torch::kInt8)),
      true);
}

TEST_F(LazyCompareKernelTest, TypePromotion) {
  auto typetest = [](at::Tensor (*op)(const at::Tensor&, const at::Tensor&),
                     c10::ScalarType dtype1,
                     c10::ScalarType dtype2,
                     c10::IntArrayRef size) {
    auto a = torch::randn(size).to(dtype1);
    auto b = torch::randn(size).to(dtype2);
    auto out = op(a, b);

    auto ha = a.to("hpu");
    auto hb = b.to("hpu");
    auto hout = op(ha, hb);
    EXPECT_TRUE(allclose(out.to(kFloat), hout.to("cpu").to(kFloat)));
  };
  // Do not create 2 or more tests with same operator and shapes, those will
  // fail when running back 2 back because lowering cache does not check for
  // dependencies within graph
  typetest(&torch::eq, torch::kFloat, torch::kLong, {3, 4});
  typetest(&torch::eq, torch::kLong, torch::kFloat, {2, 4});
  typetest(&torch::ne, torch::kInt8, torch::kInt, {2, 4});
  typetest(&torch::gt, torch::kLong, torch::kFloat, {2, 4});
  typetest(&torch::lt, torch::kLong, torch::kFloat, {2, 4});
  typetest(&torch::ge, torch::kLong, torch::kFloat, {2, 4});
}

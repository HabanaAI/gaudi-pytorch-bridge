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

class LazyCompareKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyCompareKernelTest, LtScalarTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);

  torch::Tensor A = torch::rand({2, 2}, torch::requires_grad(false));
  float compVal = 1.1f;
  auto out_cpu = torch::lt(A, compVal);

  auto hA = A.to(torch::kHABANA);
  auto result = torch::lt(hA, compVal);
  torch::Tensor out_hpu = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(out_cpu.to(torch::kFloat), out_hpu.to(torch::kFloat)), true);

  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyCompareKernelTest, LtTensorTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);

  const std::vector<int64_t> dimentions{5, 3, 4};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);

  auto expected = torch::lt(A, B);
  auto hA = A.to(torch::kHABANA);
  auto hB = B.to(torch::kHABANA);

  auto result = torch::lt(hA, hB);
  torch::Tensor habanaGenerated = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(expected.to(torch::kInt8), habanaGenerated.to(torch::kInt8)),
      true);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyCompareKernelTest, GeScalarTest) {
  torch::Tensor A = torch::rand({2, 2}, torch::requires_grad(false));
  float compVal = 1.1f;
  auto out_cpu = torch::ge(A, compVal);

  auto hA = A.to(torch::kHABANA);
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
  auto hA = A.to(torch::kHABANA);
  auto hB = B.to(torch::kHABANA);

  auto result = torch::ge(hA, hB);
  torch::Tensor habanaGenerated = result.to(torch::kCPU);

  EXPECT_EQ(
      allclose(expected.to(torch::kInt8), habanaGenerated.to(torch::kInt8)),
      true);
}
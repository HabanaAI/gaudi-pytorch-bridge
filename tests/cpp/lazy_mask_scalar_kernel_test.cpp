#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;
using namespace at;

class LazyMaskKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyMaskKernelTest, MaskedScaleInplaceTest) {
  const std::vector<int64_t> dimentions{7, 3, 5};
  const int randomLimit = 300;
  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);

  // Generate random number for scalar
  float x = (float)rand() / (float)(RAND_MAX / randomLimit);
  double scale = rand() % 2 ? x : -1 * x;

  // Eager section:
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hExpected = masked_scale_hpu(hA, hB, scale);
  Tensor expected = hExpected.to(torch::kCPU);

  // Lazy Section
  auto hAL = A.to(torch::kHPU);
  auto hBL = B.to(torch::kHPU);
  auto hOut = _masked_scale(hAL, hBL, scale);
  Tensor out = hOut.to(kCPU);

  EXPECT_EQ(allclose(out, expected), true);
}

TEST_F(LazyMaskKernelTest, MaskedFillInplaceTest) {
  const std::vector<int64_t> dimentions{3, 3};
  torch::Tensor A = torch::randn(dimentions);
  int data[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  torch::Tensor mask = torch::from_blob(data, dimentions).to(torch::kInt);

  torch::Tensor value = torch::randn({}); // Only 0-dim tensor accesped
  auto hA = A.to(torch::kHPU);
  auto cpuOut = A.masked_fill_(mask, value);

  auto hValue = value.to(torch::kHPU);
  auto hMask = mask.to(torch::kHPU);

  auto result = hA.masked_fill_(hMask, hValue);
  Tensor hOut = result.to(kCPU);
  EXPECT_TRUE(allclose(hOut, cpuOut));
}

TEST_F(LazyMaskKernelTest, MaskedFillScalarInplaceTest) {
  const std::vector<int64_t> dimentions{3, 3};
  torch::Tensor A = torch::randn(dimentions);
  int data[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  torch::Tensor mask = torch::from_blob(data, dimentions).to(torch::kInt);
  Scalar value = 35;

  auto hA = A.to(torch::kHPU);
  auto cpuOut = A.masked_fill_(mask, value);

  auto hMask = mask.to(torch::kHPU);
  auto result = hA.masked_fill_(hMask, value);
  Tensor hOut = result.to(kCPU);
  EXPECT_TRUE(allclose(hOut, cpuOut));
}

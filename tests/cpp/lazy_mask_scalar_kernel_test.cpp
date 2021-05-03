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

class LazyMaskScalarKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyMaskScalarKernelTest, MaskedScaleInplaceTest) {
  const std::vector<int64_t> dimentions{7, 3, 5};
  const int randomLimit = 300;
  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);

  // Generate random number for scalar
  float x = (float)rand() / (float)(RAND_MAX / randomLimit);
  double scale = rand() % 2 ? x : -1 * x;

  // Eager section:
  auto hA = A.to(torch::kHABANA);
  auto hB = B.to(torch::kHABANA);
  auto hExpected = masked_scale_hpu(hA, hB, scale);
  Tensor expected = hExpected.to(torch::kCPU);

  // Lazy Section
  auto hAL = A.to(torch::kHABANA);
  auto hBL = B.to(torch::kHABANA);
  auto hOut = _masked_scale(hAL, hBL, scale);
  Tensor out = hOut.to(kCPU);

  EXPECT_EQ(allclose(out, expected), true);
}

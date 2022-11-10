#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;
using namespace at;

class LazyTensorAddKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyTensorAddKernelTest, AddcmulOut) {
  const Scalar alpha = 1.5;
  const std::vector<int64_t> dimentions{3, 5, 4};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);
  torch::Tensor C = torch::randn(dimentions);
  torch::Tensor out_cpu = at::empty_like(A);

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);
  torch::Tensor out_hpu = at::empty_like(hA);

  auto result = at::addcmul_outf(hA, hB, hC, alpha, out_hpu);
  Tensor hOut = result.to(kCPU);

  auto cpuOut = at::addcmul_outf(A, B, C, alpha, out_cpu);

  EXPECT_EQ(allclose(hOut, cpuOut, COMMON_ATOL_FLOAT, COMMON_RTOL_FLOAT), true);
}

TEST_F(LazyTensorAddKernelTest, AddcdivOut) {
  const Scalar alpha = 1.6;
  const std::vector<int64_t> dimentions{4, 5, 3};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);
  torch::Tensor C = torch::randn(dimentions);
  torch::Tensor out_cpu = at::empty_like(A);

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);
  torch::Tensor out_hpu = at::empty_like(hA);

  auto result = at::addcdiv_outf(hA, hB, hC, alpha, out_hpu);
  Tensor hOut = result.to(kCPU);

  auto cpuOut = at::addcdiv_outf(A, B, C, alpha, out_cpu);

  EXPECT_EQ(allclose(hOut, cpuOut, COMMON_ATOL_FLOAT, COMMON_RTOL_FLOAT), true);
}

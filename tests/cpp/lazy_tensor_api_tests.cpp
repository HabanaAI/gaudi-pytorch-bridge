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

class LazyTensorAPITest : public habana_lazy_test::LazyTest {};

TEST_F(LazyTensorAPITest, NumelDimSizeTest) {
  torch::Tensor A = torch::tensor(
      {{{2, 4, 1, 3, 3}, {0, 9, 8, 7, 6}, {7, 7, 7, 8, 8}},
       {{2, 4, 1, 3, 3}, {9, 9, 1, 3, -2}, {8, 3, 2, 1, 0}}});
  torch::Tensor B = torch::tensor(
      {{{2, 6, 1, 1, 0}, {9, 2, 5, 6, -5}, {8, 5, 2, 1, 7}},
       {{1, 5, 1, 5, 1}, {1, 4, 1, 3, -2}, {1, 6, 8, 9, 10}}});
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor out = torch::mul(hA, hB);

  ASSERT_TRUE(out.numel() == 30);
  ASSERT_TRUE(out.dim() == 3);
  ASSERT_TRUE(out.size(0) == 2);
  ASSERT_TRUE(out.size(1) == 3);
  ASSERT_TRUE(out.size(2) == 5);
}

TEST_F(LazyTensorAPITest, EmptyStorage) {
  auto dummy = torch::ones(1).to("hpu");
  auto a = torch::empty(4, "hpu");
  habana_lazy::HbLazyTensor::StepMarker("hpu");
  ASSERT_TRUE(GetHbLazyTensor(a).CurrentTensorData() != nullopt);
}

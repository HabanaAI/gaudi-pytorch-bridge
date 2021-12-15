#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy_test_infra.h"
#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

using namespace habana_lazy;

class LazyDynamicFallbackTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyDynamicFallbackTest, FallbackCatTest) {
  bool refine_enabled = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  if (!refine_enabled) {
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES, true, 1);
  }

  int H = 4;
  std::vector<int> in_sizes{8, 16, 32};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({W}).to(torch::kInt32);
    torch::Tensor B = torch::randn({H}).to(torch::kInt32);
    torch::Tensor C = torch::randn({H + W}).to(torch::kInt32);
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hC = C.to(torch::kHPU);
    torch::Tensor cat_out = torch::cat({A, B});
    torch::Tensor h_cat_out = torch::cat({hA, hB});

    torch::Tensor hOut = torch::add(hC, h_cat_out);
    torch::Tensor out = torch::add(C, cat_out);
    EXPECT_EQ(allclose(hOut.to(torch::kCPU), out, 0.001, 0.001), true);
  }

  if (!refine_enabled) {
    UNSET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  }
}
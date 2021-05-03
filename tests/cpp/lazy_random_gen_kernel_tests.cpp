#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;

class LazyRandomGenKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyRandomGenKernelTest, FusedDropoutTest) {
  auto in = torch::randn({2, 3, 4}, torch::dtype(torch::kFloat));
  constexpr double p = 0.3;

  auto h_in = in.to(torch::kHABANA);
  auto eager_results = fused_dropout_hpu(h_in, p);
  auto eager_result1 = std::get<0>(eager_results).to("cpu");
  auto eager_result2 = std::get<1>(eager_results).to("cpu");

  auto lazy_h_in = in.to(torch::kHABANA);
  auto lazy_results = torch::_fused_dropout(lazy_h_in, p);

  auto lResult1 = std::get<0>(lazy_results).to("cpu");
  auto lResult2 = std::get<1>(lazy_results).to("cpu");

  // EXPECT_TRUE(allclose(eager_result1, lResut1, 0.01, 0.01));
  // EXPECT_TRUE(allclose(eager_result2, lResut2, 0.01, 0.01));
}

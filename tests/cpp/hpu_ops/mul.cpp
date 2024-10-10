#include <gtest/gtest-param-test.h>
#include "habana_kernels/fallback_helper.h"
#include "util.h"

class MulDtypeSupportTest : public DTypeSupportTest<c10::ScalarType> {};

TEST_P(MulDtypeSupportTest, MulOutDtypeSupportTest) {
  auto dtype = GetParam();
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kHPU);
  auto input = torch::tensor({1.2, 2.0}, options);
  auto other = torch::clone(input);
  auto output = torch::empty({2}, options);

  torch::mul_out(input, other, output);

  const auto& op_fallback_frequency =
      habana::HpuFallbackHelper::get()->get_op_count();
  EXPECT_EQ(
      op_fallback_frequency.find("aten::mul.out"), op_fallback_frequency.end());
}

INSTANTIATE_TEST_SUITE_P(
    MulOutFallback,
    MulDtypeSupportTest,
    testing::Values(
        torch::kBFloat16,
        torch::kFloat32,
        torch::kInt32,
        torch::kInt64,
        torch::kInt8,
        torch::kInt16));

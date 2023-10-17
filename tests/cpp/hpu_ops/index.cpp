/******************************************************************************
 * Copyright (C) 2023 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <gtest/gtest-param-test.h>

#include "habana_kernels/fallback_helper.h"
#include "util.h"

class IndexDTypeSupportTest : public DTypeSupportTest<c10::ScalarType> {};

TEST_P(IndexDTypeSupportTest, IndexTensorOutTest) {
  auto dtype = GetParam();
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kHPU);
  auto input = torch::tensor({{1, 2, 3, 4}, {1, 2, 3, 4}}, options);
  auto output = torch::empty({}, options);
  auto indices = torch::tensor({{0}, {1}}, options.dtype(torch::kInt64));

  torch::index_out(output, input, {indices});
  const auto& op_fallback_frequency =
      habana::HpuFallbackHelper::get()->get_op_count();
  EXPECT_EQ(
      op_fallback_frequency.find("aten::index.Tensor_out"),
      op_fallback_frequency.end());
}

INSTANTIATE_TEST_SUITE_P(
    IndexTensorOutFallback,
    IndexDTypeSupportTest,
    testing::Values(
        torch::kBFloat16,
        torch::kFloat32,
        torch::kInt32,
        torch::kInt64,
        torch::kInt8,
        torch::kUInt8));

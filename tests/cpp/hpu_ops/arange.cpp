/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <gtest/gtest-typed-test.h>
#include "habana_kernels/fallback_helper.h"
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, arange_start_out) {
  constexpr float start = 1.2;
  constexpr float end = 6.3;
  constexpr float step = 0.8;
  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::arange_outf(start, end, step, expected);
  torch::arange_outf(start, end, step, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, arange_out) {
  constexpr float end = 6.0;
  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::arange_outf(end, expected);
  torch::arange_outf(end, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, arange) {
  constexpr float end = 6.0;
  auto expected = torch::arange(end);
  auto result = torch::arange(end, torch::TensorOptions().device("hpu"));

  Compare(expected, result);
}

TEST_F(HpuOpTest, arange_start) {
  constexpr float start = 1.2;
  constexpr float end = 8.0;
  auto expected = torch::arange(start, end);
  auto result = torch::arange(start, end, torch::TensorOptions().device("hpu"));

  Compare(expected, result);
}

TEST_F(HpuOpTest, arange_start_step) {
  constexpr float start = 1.2;
  constexpr float end = 12.0;
  constexpr float step = 1.8;
  auto expected = torch::arange(start, end, step);
  auto result =
      torch::arange(start, end, step, torch::TensorOptions().device("hpu"));
  Compare(expected, result);
}

TEST_F(HpuOpTest, arange_start_step_empty_result) {
  constexpr float start = 1.2;
  constexpr float end = 2.0;
  constexpr float step = 10.8;
  auto expected = torch::arange(start, end, step);
  auto result =
      torch::arange(start, end, step, torch::TensorOptions().device("hpu"));

  Compare(expected, result);
}

TEST_F(HpuOpTest, arange_start_step_bFloat16) {
  constexpr float start = 1;
  constexpr float end = 4.0;
  constexpr float step = 1.2;
  const auto tensorOptions = torch::TensorOptions(torch::kBFloat16);
  auto expected = torch::arange(start, end, step, tensorOptions);
  auto result = torch::arange(start, end, step, tensorOptions.device("hpu"));

  Compare(expected, result);
}

TEST_F(HpuOpTest, arange_start_step_float32) {
  constexpr float start = 1;
  constexpr float end = 5.0;
  constexpr float step = 2.2;
  const auto tensorOptions = torch::TensorOptions(torch::kFloat);
  auto expected = torch::arange(start, end, step, tensorOptions);
  auto result = torch::arange(start, end, step, tensorOptions.device("hpu"));

  Compare(expected, result);
}

TEST_F(HpuOpTest, arange_start_step_long) {
  constexpr int start = 1;
  constexpr int end = 40;
  constexpr int step = 1;
  const auto tensorOptions = torch::TensorOptions(torch::kLong);
  auto expected = torch::arange(start, end, step, tensorOptions);
  auto result = torch::arange(start, end, step, tensorOptions.device("hpu"));

  Compare(expected, result);
}

TEST_F(HpuOpTest, arange_start_step_mixed_dtypes) {
  constexpr int start = 1;
  constexpr int end = 10;
  constexpr float step = 1.5;
  auto expected = torch::arange(start, end, step);
  auto result =
      torch::arange(start, end, step, torch::TensorOptions().device("hpu"));

  Compare(expected, result);
}

class ArangeDTypeSupportTest : public DTypeSupportTest<c10::ScalarType> {};

TEST_P(ArangeDTypeSupportTest, ArangeStartOutDTypeSupportTest) {
  auto dtype = GetParam();
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kHPU);
  auto out = torch::empty({10}, options);

  torch::arange_outf(0.0, 1.0, 10.0, out);

  const auto& op_fallback_frequency =
      habana::HpuFallbackHelper::get()->get_op_count();
  EXPECT_EQ(
      op_fallback_frequency.find("aten:arange.start_out"),
      op_fallback_frequency.end());
}

INSTANTIATE_TEST_SUITE_P(
    ArangeStartOutFallback,
    ArangeDTypeSupportTest,
    testing::Values(
        torch::kBFloat16,
        torch::kFloat32,
        torch::kInt32,
        torch::kInt64,
        torch::kInt8,
        torch::kInt16));

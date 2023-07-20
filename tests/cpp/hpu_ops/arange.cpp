/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

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

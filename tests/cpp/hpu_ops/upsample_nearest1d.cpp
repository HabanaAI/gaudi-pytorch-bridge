/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, upsample_nearest1d_fwd_scale) {
  GenerateInputs(1, {{2, 3, 4}});
  std::vector<double> scale_factor = {0.999};

  auto expected =
      torch::upsample_nearest1d(GetCpuInput(0), c10::nullopt, scale_factor);
  auto result =
      torch::upsample_nearest1d(GetHpuInput(0), c10::nullopt, scale_factor);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest1d_fwd_size) {
  GenerateInputs(1, {{2, 3, 4}});
  std::vector<int64_t> size = {6};

  auto expected = torch::upsample_nearest1d(GetCpuInput(0), size);
  auto result = torch::upsample_nearest1d(GetHpuInput(0), size);

  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest1d_fwd_out) {
  GenerateInputs(1, {{5, 6, 4}});

  torch::ScalarType dtype = torch::kFloat;
  std::vector<int64_t> size = {7};

  auto expected = torch::empty({5, 6, 7}, dtype);
  auto result =
      torch::empty({5, 6, 7}, torch::TensorOptions(dtype).device("hpu"));
  c10::optional<double> scales = c10::nullopt;

  torch::upsample_nearest1d_outf(GetCpuInput(0), size, scales, expected);
  torch::upsample_nearest1d_outf(GetHpuInput(0), size, scales, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest1d_fwd_usual) {
  GenerateInputs(1, {{4, 3, 25}});
  std::vector<int64_t> size = {4};
  c10::optional<double> scales = 3.0;

  auto expected = torch::upsample_nearest1d(GetCpuInput(0), size, scales);
  auto result = torch::upsample_nearest1d(GetHpuInput(0), size, scales);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest1d_bwd_size) {
  GenerateInputs(1, {{1, 4, 6}});
  std::vector<int64_t> out_size = {6};
  std::vector<int64_t> input_size = {1, 4, 3};

  auto expected =
      torch::upsample_nearest1d_backward(GetCpuInput(0), out_size, input_size);
  auto result =
      torch::upsample_nearest1d_backward(GetHpuInput(0), out_size, input_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest1d_bwd_scale) {
  GenerateInputs(1, {{10, 5, 12}});
  c10::optional<double> scale_factor(3.0);
  std::vector<int64_t> input_size = {10, 5, 4};
  std::vector<int64_t> output_size = {12};

  auto expected = torch::upsample_nearest1d_backward(
      GetCpuInput(0), output_size, input_size, scale_factor);
  auto result = torch::upsample_nearest1d_backward(
      GetHpuInput(0), output_size, input_size, scale_factor);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest1d_bwd_out) {
  GenerateInputs(1, {{2, 28, 64}});
  std::vector<int64_t> out_size = {64};
  std::vector<int64_t> input_size = {2, 28, 16};

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(input_size, dtype);
  auto result = expected.to(torch::kHPU);
  c10::optional<double> scales = c10::nullopt;

  torch::upsample_nearest1d_backward_outf(
      GetCpuInput(0), out_size, input_size, scales, expected);
  torch::upsample_nearest1d_backward_outf(
      GetHpuInput(0), out_size, input_size, scales, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest1d_bwd_usual) {
  GenerateInputs(1, {{1, 4, 6}});
  std::vector<int64_t> out_size = {6};
  c10::optional<double> scale_factor = 2.0;
  std::vector<int64_t> input_size = {1, 4, 3};

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(input_size, dtype);
  auto result = expected.to(torch::kHPU);

  torch::upsample_nearest1d_backward(
      GetCpuInput(0), out_size, input_size, scale_factor);
  torch::upsample_nearest1d_backward(
      GetHpuInput(0), out_size, input_size, scale_factor);
  Compare(expected, result);
}

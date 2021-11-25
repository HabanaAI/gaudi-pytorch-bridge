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

#define TENSOR_TYPE_float torch::kFloat

class HpuOpTest : public HpuOpTestUtil {};

// forward variants
TEST_F(HpuOpTest, upsample_bilinear2d_fwd_scale) {
  GenerateInputs(1, {{2, 7, 3, 4}});
  std::vector<double> scale_factor = {1.999, 2.999};

  auto expected = torch::upsample_bilinear2d(
      GetCpuInput(0), c10::nullopt, /*align_corner*/ true, scale_factor);
  auto result = torch::upsample_bilinear2d(
      GetHpuInput(0), c10::nullopt, /*align_corner*/ true, scale_factor);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_bilinear2d_fwd_size) {
  GenerateInputs(1, {{4, 5, 3, 25}});
  std::vector<int64_t> size = {8, 50};

  auto expected =
      torch::upsample_bilinear2d(GetCpuInput(0), size, /*align_corner*/ false);
  auto result =
      torch::upsample_bilinear2d(GetHpuInput(0), size, /*align_corner*/ false);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_bilinear2d_fwd_out) {
  GenerateInputs(1, {{5, 6, 8, 4}});
  std::vector<int64_t> size = {7, 9};
  auto expected = torch::empty(0, TENSOR_TYPE_float);
  auto result = expected.to(torch::kHPU);

  torch::upsample_bilinear2d_outf(
      GetCpuInput(0), size, /*align_corner*/ false, {}, {}, expected);
  torch::upsample_bilinear2d_outf(
      GetHpuInput(0), size, /*align_corner*/ false, {}, {}, result);
  Compare(expected, result);
}

// backward variants
TEST_F(HpuOpTest, upsample_bilinear2d_bwd_size) {
  GenerateInputs(1, {{4, 3, 12, 64}});
  std::vector<int64_t> output_size = {12, 64};
  std::vector<int64_t> input_size = {4, 3, 6, 32};

  auto expected = torch::upsample_bilinear2d_backward(
      GetCpuInput(0), output_size, input_size, /*align_corner*/ true);
  auto result = torch::upsample_bilinear2d_backward(
      GetHpuInput(0), output_size, input_size, /*align_corner*/ true);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_bilinear2d_bwd_scale) {
  GenerateInputs(1, {{1, 2, 6, 12}});
  std::vector<double> scale = {2.0, 3.0};
  std::vector<int64_t> input_size = {1, 2, 3, 4};

  auto expected = torch::upsample_bilinear2d_backward(
      GetCpuInput(0), c10::nullopt, input_size, /*align_corner*/ false, scale);
  auto result = torch::upsample_bilinear2d_backward(
      GetHpuInput(0), c10::nullopt, input_size, /*align_corner*/ false, scale);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_bilinear2d_bwd_out) {
  GenerateInputs(1, {{1, 5, 28, 64}});
  std::vector<int64_t> output_size = {28, 64};
  std::vector<int64_t> input_size = {1, 5, 28, 16};
  auto expected = torch::empty(0, TENSOR_TYPE_float);
  auto result = expected.to(torch::kHPU);

  torch::upsample_bilinear2d_backward_outf(
      GetCpuInput(0),
      output_size,
      input_size,
      /*align_corner*/ false,
      {},
      {},
      expected);
  torch::upsample_bilinear2d_backward_outf(
      GetHpuInput(0),
      output_size,
      input_size,
      /*align_corner*/ false,
      {},
      {},
      result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest2d_fwd_scale) {
  GenerateInputs(1, {{2, 7, 3, 4}});
  std::vector<int64_t> size = {6, 12};
  c10::optional<double> scale_h = 2.0;
  c10::optional<double> scale_w = 3.0;

  auto expected =
      torch::upsample_nearest2d(GetCpuInput(0), size, scale_h, scale_w);
  auto result =
      torch::upsample_nearest2d(GetHpuInput(0), size, scale_h, scale_w);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest2d_fwd_size) {
  GenerateInputs(1, {{2, 7, 3, 4}});
  std::vector<int64_t> size = {10, 17};

  auto expected = torch::upsample_nearest2d(GetCpuInput(0), size);
  auto result = torch::upsample_nearest2d(GetHpuInput(0), size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest2d_fwd_out) {
  GenerateInputs(1, {{5, 6, 4, 7}});

  torch::ScalarType dtype = torch::kFloat;
  std::vector<int64_t> size = {7, 9};

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::upsample_nearest2d_outf(GetCpuInput(0), size, {}, {}, expected);
  torch::upsample_nearest2d_outf(GetHpuInput(0), size, {}, {}, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest2d_bwd_size) {
  GenerateInputs(1, {{1, 4, 6, 8}});
  std::vector<int64_t> out_size = {6, 8};
  std::vector<int64_t> input_size = {1, 4, 3, 4};

  auto expected =
      torch::upsample_nearest2d_backward(GetCpuInput(0), out_size, input_size);
  auto result =
      torch::upsample_nearest2d_backward(GetHpuInput(0), out_size, input_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest2d_bwd_out) {
  GenerateInputs(1, {{2, 28, 64, 64}});
  std::vector<int64_t> out_size = {64, 64};
  std::vector<int64_t> input_size = {2, 28, 16, 32};

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = expected.to(torch::kHPU);

  torch::upsample_nearest2d_backward_outf(
      GetCpuInput(0),
      out_size,
      input_size,
      c10::nullopt,
      c10::nullopt,
      expected);
  torch::upsample_nearest2d_backward_outf(
      GetHpuInput(0), out_size, input_size, c10::nullopt, c10::nullopt, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest2d_bwd_usual) {
  GenerateInputs(1, {{1, 4, 6, 8}});
  std::vector<int64_t> out_size = {6, 8};
  c10::optional<double> scale_h = 2.0;
  c10::optional<double> scale_w = 2.0;
  std::vector<int64_t> input_size = {1, 4, 3, 4};

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = expected.to(torch::kHPU);

  torch::upsample_nearest2d_backward(
      GetCpuInput(0), out_size, input_size, scale_h, scale_w);
  torch::upsample_nearest2d_backward(
      GetHpuInput(0), out_size, input_size, scale_h, scale_w);
  Compare(expected, result);
}
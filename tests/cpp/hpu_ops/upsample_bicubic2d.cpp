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

#define TENSOR_TYPE_float torch::kFloat

class HpuOpTest : public HpuOpTestUtil {};

// forward variants
TEST_F(HpuOpTest, upsample_bicubic2d_fwd_scale_CL) {
  GenerateInputs(1, {{2, 7, 3, 4}});
  std::vector<double> scale_factor = {1.999, 2.999};

  auto expected = torch::upsample_bicubic2d(
      GetCpuInput(0).to(c10::MemoryFormat::ChannelsLast),
      c10::nullopt,
      /*align_corner*/ false,
      scale_factor);

  auto result = torch::upsample_bicubic2d(
      GetCpuInput(0).to(c10::MemoryFormat::ChannelsLast).to("hpu"),
      c10::nullopt,
      /*align_corner*/ false,
      scale_factor);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_bicubic2d_fwd_scale) {
  GenerateInputs(1, {{2, 7, 3, 4}});
  std::vector<double> scale_factor = {1.999, 2.999};

  auto expected = torch::upsample_bicubic2d(
      GetCpuInput(0),
      c10::nullopt,
      /*align_corner*/ false,
      scale_factor);
  auto result = torch::upsample_bicubic2d(
      GetHpuInput(0),
      c10::nullopt,
      /*align_corner*/ false,
      scale_factor);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_bicubic2d_fwd_scale_zero) {
  GenerateInputs(1, {{2, 7, 3, 4}});
  std::vector<double> scale_factor = {0.6, 0.7};

  auto expected = torch::upsample_bicubic2d(
      GetCpuInput(0), c10::nullopt, /*align_corner*/ true, scale_factor);
  auto result = torch::upsample_bicubic2d(
      GetHpuInput(0),
      c10::nullopt,
      /*align_corner*/ true,
      scale_factor);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_bicubic2d_fwd_size) {
  GenerateInputs(1, {{4, 5, 3, 25}});
  std::vector<int64_t> size = {8, 50};

  auto expected =
      torch::upsample_bicubic2d(GetCpuInput(0), size, /*align_corner*/ false);
  auto result =
      torch::upsample_bicubic2d(GetHpuInput(0), size, /*align_corner*/ false);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_bicubic2d_fwd_out) {
  GenerateInputs(1, {{5, 6, 8, 4}});
  std::vector<int64_t> size = {7, 9};
  auto expected = torch::empty(0, TENSOR_TYPE_float);
  auto result = expected.to(torch::kHPU);

  torch::upsample_bicubic2d_outf(
      GetCpuInput(0), size, /*align_corner*/ false, {}, {}, expected);
  torch::upsample_bicubic2d_outf(
      GetHpuInput(0), size, /*align_corner*/ false, {}, {}, result);
  Compare(expected, result);
}

// backward variants
TEST_F(HpuOpTest, upsample_bicubic2d_bwd_size) {
  GenerateInputs(1, {{4, 3, 12, 64}});
  std::vector<int64_t> output_size = {12, 64};
  std::vector<int64_t> input_size = {4, 3, 6, 32};

  auto expected = torch::upsample_bicubic2d_backward(
      GetCpuInput(0), output_size, input_size, /*align_corner*/ true);
  auto result = torch::upsample_bicubic2d_backward(
      GetHpuInput(0), output_size, input_size, /*align_corner*/ true);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_bicubic2d_bwd_scale_CL) {
  GenerateInputs(1, {{2, 7, 1, 6}});
  c10::optional<double> scale_h(0.6);
  c10::optional<double> scale_w(1.7);
  std::vector<int64_t> input_size = {2, 7, 3, 4};
  std::vector<int64_t> output_size = {1, 6};

  auto expected = torch::upsample_bicubic2d_backward(
      GetCpuInput(0).to(c10::MemoryFormat::ChannelsLast),
      output_size,
      input_size,
      /*align_corner*/ true,
      scale_h,
      scale_w);
  auto result = torch::upsample_bicubic2d_backward(
      GetHpuInput(0).to(c10::MemoryFormat::ChannelsLast),
      output_size,
      input_size,
      /*align_corner*/ true,
      scale_h,
      scale_w);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_bicubic2d_bwd_scale) {
  GenerateInputs(1, {{2, 7, 1, 6}});
  c10::optional<double> scale_h(0.6);
  c10::optional<double> scale_w(1.7);
  std::vector<int64_t> input_size = {2, 7, 3, 4};
  std::vector<int64_t> output_size = {1, 6};

  auto expected = torch::upsample_bicubic2d_backward(
      GetCpuInput(0),
      output_size,
      input_size,
      /*align_corner*/ true,
      scale_h,
      scale_w);
  auto result = torch::upsample_bicubic2d_backward(
      GetHpuInput(0),
      output_size,
      input_size,
      /*align_corner*/ true,
      scale_h,
      scale_w);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_bicubic2d_bwd_out_CL) {
  GenerateInputs(1, {{1, 5, 28, 64}});
  std::vector<int64_t> output_size = {28, 64};
  std::vector<int64_t> input_size = {1, 5, 28, 16};
  auto expected = torch::empty(0, TENSOR_TYPE_float);
  auto result = expected.to(torch::kHPU);

  torch::upsample_bicubic2d_backward_outf(
      GetCpuInput(0).to(c10::MemoryFormat::ChannelsLast),
      output_size,
      input_size,
      /*align_corner*/ false,
      {},
      {},
      expected);
  torch::upsample_bicubic2d_backward_outf(
      GetHpuInput(0).to(c10::MemoryFormat::ChannelsLast),
      output_size,
      input_size,
      /*align_corner*/ false,
      {},
      {},
      result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_bicubic2d_bwd_out) {
  GenerateInputs(1, {{1, 5, 28, 64}});
  std::vector<int64_t> output_size = {28, 64};
  std::vector<int64_t> input_size = {1, 5, 28, 16};
  auto expected = torch::empty(0, TENSOR_TYPE_float);
  auto result = expected.to(torch::kHPU);

  torch::upsample_bicubic2d_backward_outf(
      GetCpuInput(0),
      output_size,
      input_size,
      /*align_corner*/ false,
      {},
      {},
      expected);
  torch::upsample_bicubic2d_backward_outf(
      GetHpuInput(0),
      output_size,
      input_size,
      /*align_corner*/ false,
      {},
      {},
      result);
  Compare(expected, result);
}
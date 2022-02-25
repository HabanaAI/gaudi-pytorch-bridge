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
      GetCpuInput(0).to(c10::MemoryFormat::ChannelsLast),
      c10::nullopt,
      /*align_corner*/ false,
      scale_factor);
  auto result = torch::upsample_bilinear2d(
      GetHpuInput(0).to(c10::MemoryFormat::ChannelsLast),
      c10::nullopt,
      /*align_corner*/ false,
      scale_factor);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_bilinear2d_fwd_scale_zero) {
  GenerateInputs(1, {{2, 7, 3, 4}});
  std::vector<double> scale_factor = {0.6, 1.7};

  auto expected = torch::upsample_bilinear2d(
      GetCpuInput(0).to(c10::MemoryFormat::ChannelsLast),
      c10::nullopt,
      /*align_corner*/ true,
      scale_factor);
  auto result = torch::upsample_bilinear2d(
      GetHpuInput(0).to(c10::MemoryFormat::ChannelsLast),
      c10::nullopt,
      /*align_corner*/ true,
      scale_factor);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_bilinear2d_fwd_size) {
  GenerateInputs(1, {{4, 5, 3, 25}});
  std::vector<int64_t> size = {8, 50};

  auto expected = torch::upsample_bilinear2d(
      GetCpuInput(0).to(c10::MemoryFormat::ChannelsLast),
      size,
      /*align_corner*/ false);
  auto result = torch::upsample_bilinear2d(
      GetHpuInput(0).to(c10::MemoryFormat::ChannelsLast),
      size,
      /*align_corner*/ false);
  Compare(expected, result);
}

// Out variant failing with only zero's gets appended
// TEST_F(HpuOpTest, upsample_bilinear2d_fwd_out) {
//  GenerateInputs(1, {{5, 6, 8, 4}});
//  std::vector<int64_t> size = {7, 9};
//  auto expected = torch::empty(0, TENSOR_TYPE_float);
//  auto result = expected.to(torch::kHPU);
//
//  torch::upsample_bilinear2d_outf(
//      GetCpuInput(0).to(c10::MemoryFormat::ChannelsLast), size,
//      /*align_corner*/ false, {}, {}, expected);
//  torch::upsample_bilinear2d_outf(
//      GetHpuInput(0).to(c10::MemoryFormat::ChannelsLast), size,
//      /*align_corner*/ false, {}, {}, result);
//  Compare(expected, result);
// }

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
  GenerateInputs(1, {{2, 7, 1, 6}});
  std::vector<double> scales = {0.6, 1.7};
  std::vector<int64_t> input_size = {2, 7, 3, 4};

  auto expected = torch::upsample_bilinear2d_backward(
      GetCpuInput(0).to(c10::MemoryFormat::ChannelsLast),
      c10::nullopt,
      input_size,
      /*align_corner*/ true,
      scales);
  auto result = torch::upsample_bilinear2d_backward(
      GetHpuInput(0).to(c10::MemoryFormat::ChannelsLast),
      c10::nullopt,
      input_size,
      /*align_corner*/ true,
      scales);
  Compare(expected, result);
}

// Out Variant fails
// TEST_F(HpuOpTest, upsample_bilinear2d_bwd_out) {
//   GenerateInputs(1, {{1, 5, 28, 64}});
//   std::vector<int64_t> output_size = {28, 64};
//   std::vector<int64_t> input_size = {1, 5, 28, 16};
//   auto expected = torch::empty(0, TENSOR_TYPE_float);
//   auto result = expected.to(torch::kHPU);
//
//   torch::upsample_bilinear2d_backward_outf(
//       GetCpuInput(0),
//       output_size,
//       input_size,
//       /*align_corner*/ false,
//       {},
//       {},
//       expected);
//   torch::upsample_bilinear2d_backward_outf(
//       GetHpuInput(0),
//       output_size,
//       input_size,
//       /*align_corner*/ false,
//       {},
//       {},
//       result);
//   Compare(expected, result);
// }
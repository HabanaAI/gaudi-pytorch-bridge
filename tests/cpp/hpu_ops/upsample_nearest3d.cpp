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

TEST_F(HpuOpTest, upsample_nearest3d_fwd_out) {
  GenerateInputs(1, {{2, 7, 3, 4, 5}});
  std::vector<int64_t> size = {6, 12, 10};
  c10::optional<double> scale_h = 6.0;
  c10::optional<double> scale_w = 4.0;
  c10::optional<double> scale_d = 4.0;
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::upsample_nearest3d_outf(
      GetCpuInput(0), size, scale_d, scale_h, scale_w, expected);
  torch::upsample_nearest3d_outf(
      GetHpuInput(0), size, scale_d, scale_h, scale_w, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest3d_fwd_scale) {
  GenerateInputs(1, {{2, 7, 3, 4, 5}}, torch::kByte);
  std::vector<double> scale_factor = {0.99, 0.7, 1.2};

  auto expected = torch::upsample_nearest3d(GetCpuInput(0), {}, scale_factor);
  auto result = torch::upsample_nearest3d(GetHpuInput(0), {}, scale_factor);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest3d_bwd_size) {
  GenerateInputs(1, {{1, 4, 6, 8, 4}});
  std::vector<int64_t> out_size = {6, 8, 4};
  c10::optional<double> scale_d = 2.0;
  c10::optional<double> scale_h = 2.0;
  c10::optional<double> scale_w = 1.0;
  std::vector<int64_t> input_size = {1, 4, 3, 4, 4};

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = expected.to(torch::kHPU);

  expected = torch::upsample_nearest3d_backward(
      GetCpuInput(0), out_size, input_size, scale_d, scale_h, scale_w);
  result = torch::upsample_nearest3d_backward(
      GetHpuInput(0), out_size, input_size, scale_d, scale_h, scale_w);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_nearest3d_bwd_out) {
  GenerateInputs(1, {{2, 4, 10, 20, 30}});
  std::vector<int64_t> out_size = {10, 20, 30};
  std::vector<int64_t> input_size = {2, 4, 5, 4, 10};

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = expected.to(torch::kHPU);

  torch::upsample_nearest3d_backward_outf(
      GetCpuInput(0), out_size, input_size, {}, {}, {}, expected);
  torch::upsample_nearest3d_backward_outf(
      GetHpuInput(0), out_size, input_size, {}, {}, {}, result);
  Compare(expected, result);
}

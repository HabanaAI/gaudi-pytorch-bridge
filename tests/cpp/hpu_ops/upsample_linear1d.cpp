/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "util.h"

#define TENSOR_TYPE_float torch::kFloat

class HpuOpTest : public HpuOpTestUtil {};

// forward variants
TEST_F(HpuOpTest, upsample_linear1d_fwd_scale) {
  GenerateInputs(1, {{1, 3, 100}});
  std::vector<double> scale_factor = {0.123};

  auto expected = torch::upsample_linear1d(
      GetCpuInput(0), c10::nullopt, /*align_corner*/ true, scale_factor);
  auto result = torch::upsample_linear1d(
      GetHpuInput(0), c10::nullopt, /*align_corner*/ true, scale_factor);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_linear1d_fwd_size) {
  GenerateInputs(1, {{4, 3, 25}});
  std::vector<int64_t> size = {50};
  c10::optional<double> scales = 3.0;

  auto expected = torch::upsample_linear1d(
      GetCpuInput(0), size, /*align_corner*/ false, scales);
  auto result = torch::upsample_linear1d(
      GetHpuInput(0), size, /*align_corner*/ false, scales);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_linear1d_fwd_out) {
  GenerateInputs(1, {{5, 6, 4}});
  std::vector<int64_t> size = {7};
  auto expected = torch::empty({5, 6, 7}, TENSOR_TYPE_float);
  auto result = expected.to(torch::kHPU);
  c10::optional<double> scales = c10::nullopt;

  torch::upsample_linear1d_outf(
      GetCpuInput(0), size, /*align_corner*/ false, scales, expected);
  torch::upsample_linear1d_outf(
      GetHpuInput(0), size, /*align_corner*/ false, scales, result);
  Compare(expected, result);
}

// backward variants
TEST_F(HpuOpTest, upsample_linear1d_bwd_size) {
  GenerateInputs(1, {{4, 3, 64}});
  std::vector<int64_t> output_size = {64};
  std::vector<int64_t> input_size = {4, 3, 32};
  c10::optional<double> scales = {2.0};

  auto expected = torch::upsample_linear1d_backward(
      GetCpuInput(0), output_size, input_size, /*align_corner*/ true, scales);
  auto result = torch::upsample_linear1d_backward(
      GetHpuInput(0), output_size, input_size, /*align_corner*/ true, scales);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_linear1d_bwd_scale) {
  GenerateInputs(1, {{2, 3, 8}});
  c10::optional<double> scale(2.0);
  std::vector<int64_t> output_size = {8};
  std::vector<int64_t> input_size = {2, 3, 4};

  auto expected = torch::upsample_linear1d_backward(
      GetCpuInput(0), output_size, input_size, /*align_corner*/ false, scale);
  auto result = torch::upsample_linear1d_backward(
      GetHpuInput(0), output_size, input_size, /*align_corner*/ false, scale);
  Compare(expected, result);
}

TEST_F(HpuOpTest, upsample_linear1d_bwd_out) {
  GenerateInputs(1, {{1, 28, 64}});
  std::vector<int64_t> output_size = {64};
  std::vector<int64_t> input_size = {1, 28, 16};
  auto expected = torch::empty(input_size, TENSOR_TYPE_float);
  auto result = expected.to(torch::kHPU);
  c10::optional<double> scales = c10::nullopt;

  torch::upsample_linear1d_backward_outf(
      GetCpuInput(0),
      output_size,
      input_size,
      /*align_corner*/ false,
      scales,
      expected);
  torch::upsample_linear1d_backward_outf(
      GetHpuInput(0),
      output_size,
      input_size,
      /*align_corner*/ false,
      scales,
      result);
  Compare(expected, result);
}

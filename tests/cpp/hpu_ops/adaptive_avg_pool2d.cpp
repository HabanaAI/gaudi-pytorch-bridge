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

class HpuOpTest : public HpuOpTestUtil {
 public:
  void TestAdaptiveAvgPool2d(
      torch::ArrayRef<torch::IntArrayRef> input_shapes,
      const std::vector<int64_t>& output_size,
      torch::ScalarType dtype,
      bool is_out = false) {
    GenerateInputs(1, input_shapes, {dtype});
    double rtol = 1e-03;
    double atol = 1e-03;
    if (dtype == torch::kBFloat16) {
      rtol = 1e-02;
      atol = 1e-02;
    }
    if (is_out) {
      std::vector<int64_t> output_size_with_batch = input_shapes[0].vec();
      std::copy_n(
          output_size.begin(),
          output_size.size(),
          output_size_with_batch.end() - output_size.size());
      auto expected = torch::empty(output_size_with_batch).to(dtype);
      auto result = torch::empty(output_size_with_batch, "hpu").to(dtype);
      torch::_adaptive_avg_pool2d_outf(GetCpuInput(0), output_size, expected);
      torch::_adaptive_avg_pool2d_outf(GetHpuInput(0), output_size, result);
      Compare(expected, result, rtol, atol);
    } else {
      auto expected = torch::_adaptive_avg_pool2d(GetCpuInput(0), output_size);
      auto result = torch::_adaptive_avg_pool2d(GetHpuInput(0), output_size);
      Compare(expected, result, rtol, atol);
    }
  }

  void TestAdaptiveAvgPool2dBwd(
      torch::ArrayRef<torch::IntArrayRef> input_shapes,
      torch::ScalarType dtype,
      bool is_out = false) {
    GenerateInputs(2, input_shapes, {dtype, dtype});
    double rtol = 1e-03;
    double atol = 1e-03;
    if (dtype == torch::kBFloat16) {
      rtol = 1e-02;
      atol = 1e-02;
    }
    if (is_out) {
      auto expected = torch::empty(input_shapes[1]).to(dtype);
      auto result = torch::empty(input_shapes[1], "hpu").to(dtype);
      // op flavor not existent in 1.12
      torch::_adaptive_avg_pool2d_backward_outf(
          GetCpuInput(0), GetCpuInput(1), expected);
      torch::_adaptive_avg_pool2d_backward_outf(
          GetHpuInput(0), GetHpuInput(1), result);
      Compare(expected, result, rtol, atol);
    } else {
      auto expected =
          torch::_adaptive_avg_pool2d_backward(GetCpuInput(0), GetCpuInput(1));
      auto result =
          torch::_adaptive_avg_pool2d_backward(GetHpuInput(0), GetHpuInput(1));
      Compare(expected, result, rtol, atol);
    }
  }
};

TEST_F(HpuOpTest, adaptive_avg_pool2d_f32_batch) {
  TestAdaptiveAvgPool2d({{8, 3, 27, 27}}, {3, 3}, torch::kFloat);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_bf16_batch) {
  TestAdaptiveAvgPool2d({{8, 3, 27, 27}}, {3, 3}, torch::kBFloat16);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_f32_no_batch) {
  TestAdaptiveAvgPool2d({{4, 15, 10}}, {5, 4}, torch::kFloat);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_bf16_no_batch) {
  TestAdaptiveAvgPool2d({{4, 15, 10}}, {5, 4}, torch::kBFloat16);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_out_f32_batch) {
  TestAdaptiveAvgPool2d({{8, 3, 27, 27}}, {3, 3}, torch::kFloat, true);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_out_bf16_batch) {
  TestAdaptiveAvgPool2d({{8, 3, 27, 27}}, {3, 3}, torch::kBFloat16, true);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_out_f32_no_batch) {
  TestAdaptiveAvgPool2d({{4, 15, 10}}, {5, 4}, torch::kFloat, true);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_out_bf16_no_batch) {
  TestAdaptiveAvgPool2d({{4, 15, 10}}, {5, 4}, torch::kBFloat16, true);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_backward_f32_batch) {
  TestAdaptiveAvgPool2dBwd({{8, 3, 3, 3}, {8, 3, 27, 27}}, torch::kFloat);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_backward_bf16_batch) {
  TestAdaptiveAvgPool2dBwd({{8, 3, 3, 3}, {8, 3, 27, 27}}, torch::kBFloat16);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_backward_f32_no_batch) {
  TestAdaptiveAvgPool2dBwd({{4, 5, 4}, {4, 15, 10}}, torch::kFloat);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_backward_bf16_no_batch) {
  TestAdaptiveAvgPool2dBwd({{4, 5, 4}, {4, 15, 10}}, torch::kBFloat16);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_backward_out_f32_batch) {
  TestAdaptiveAvgPool2dBwd({{8, 3, 3, 3}, {8, 3, 27, 27}}, torch::kFloat, true);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_backward_out_bf16_batch) {
  TestAdaptiveAvgPool2dBwd(
      {{8, 3, 3, 3}, {8, 3, 27, 27}}, torch::kBFloat16, true);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_backward_out_f32_no_batch) {
  TestAdaptiveAvgPool2dBwd({{4, 5, 4}, {4, 15, 10}}, torch::kFloat, true);
}

TEST_F(HpuOpTest, adaptive_avg_pool2d_backward_out_bf16_no_batch) {
  TestAdaptiveAvgPool2dBwd({{4, 5, 4}, {4, 15, 10}}, torch::kBFloat16, true);
}

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

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, LeakyRelu_bwd_out) {
  GenerateInputs(2);
  float neg_slope = GenerateScalar<float>(1e-2, 1e2);
  bool self_is_result = false;

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::leaky_relu_backward_outf(
      GetCpuInput(0), GetCpuInput(1), neg_slope, self_is_result, expected);
  torch::leaky_relu_backward_outf(
      GetHpuInput(0), GetHpuInput(1), neg_slope, self_is_result, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, LeakyRelu_bwd_out_bf16) {
  GenerateInputs(2, {torch::kBFloat16});
  float neg_slope = GenerateScalar<float>(1e-2, 1e2);
  bool self_is_result = false;

  torch::ScalarType dtype = torch::kBFloat16;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::leaky_relu_backward_outf(
      GetCpuInput(0), GetCpuInput(1), neg_slope, self_is_result, expected);
  torch::leaky_relu_backward_outf(
      GetHpuInput(0), GetHpuInput(1), neg_slope, self_is_result, result);
  Compare(expected, result, 0.01, 0.01);
}
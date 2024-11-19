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

// hardshrink_cpu not implemented for 'BFloat16'
TEST_F(HpuOpTest, hardshrink) {
  GenerateInputs(1);
  float lambda = GenerateScalar<float>();
  auto expected = torch::hardshrink(GetCpuInput(0), lambda);
  auto result = torch::hardshrink(GetHpuInput(0), lambda);
  Compare(expected, result);
}

TEST_F(HpuOpTest, hardshrink_out) {
  GenerateInputs(2);
  float lambda = GenerateScalar<float>();

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  result = torch::hardshrink_outf(GetHpuInput(0), lambda, result);
  expected = torch::hardshrink_outf(GetCpuInput(0), lambda, expected);

  Compare(expected, result);
}

TEST_F(HpuOpTest, hardshrink_backward) {
  GenerateInputs(2);
  float lambda = GenerateScalar<float>();
  auto expected =
      torch::hardshrink_backward(GetCpuInput(0), GetCpuInput(1), lambda);
  auto result =
      torch::hardshrink_backward(GetHpuInput(0), GetHpuInput(1), lambda);
  Compare(expected, result);
}

TEST_F(HpuOpTest, hardshrink_backward_out) {
  GenerateInputs(2);
  float lambda = GenerateScalar<float>();

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  expected = torch::hardshrink_backward_outf(
      GetCpuInput(1), GetCpuInput(0), lambda, expected);
  result = torch::hardshrink_backward_outf(
      GetHpuInput(1), GetHpuInput(0), lambda, result);

  Compare(expected, result);
}

// softshrink_cpu not implemented for 'BFloat16'
TEST_F(HpuOpTest, softshrink) {
  GenerateInputs(1);
  float lambda = GenerateScalar<float>();
  auto expected = torch::softshrink(GetCpuInput(0), lambda);
  auto result = torch::softshrink(GetHpuInput(0), lambda);
  Compare(expected, result);
}

TEST_F(HpuOpTest, softshrink_backward) {
  GenerateInputs(2);
  float lambda = GenerateScalar<float>();
  auto expected =
      torch::softshrink_backward(GetCpuInput(0), GetCpuInput(1), lambda);
  auto result =
      torch::softshrink_backward(GetHpuInput(0), GetHpuInput(1), lambda);
  Compare(expected, result);
}

TEST_F(HpuOpTest, softshrink_out) {
  GenerateInputs(1);
  float lambda = GenerateScalar<float>();
  auto expected = torch::empty(0, torch::kFloat32);
  auto result = expected.to(torch::kHPU);
  torch::softshrink_outf(GetCpuInput(0), lambda, expected);
  torch::softshrink_outf(GetHpuInput(0), lambda, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, softshrink_backward_out) {
  GenerateInputs(2);
  float lambda = GenerateScalar<float>();
  auto expected = torch::empty(0, torch::kFloat32);
  auto result = expected.to(torch::kHPU);
  torch::softshrink_backward_outf(
      GetCpuInput(0), GetCpuInput(1), lambda, expected);
  torch::softshrink_backward_outf(
      GetHpuInput(0), GetHpuInput(1), lambda, result);
  Compare(expected, result);
}
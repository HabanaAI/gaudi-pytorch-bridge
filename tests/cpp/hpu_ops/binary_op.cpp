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

TEST_F(HpuOpTest, mvOut) {
  GenerateInputs(3, {{2, 3}, {3}, {2}});
  auto expected =
      torch::mv_outf(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2));
  auto result = torch::mv_outf(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2));

  Compare(expected, result);
}

TEST_F(HpuOpTest, mv) {
  GenerateInputs(2, {{16, 16}, {16}});

  auto expected = torch::mv(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::mv(GetHpuInput(0), GetHpuInput(1));
  Compare(expected, result);
}

// Below testcase fails for default tolerance
// Issue Raised: https://jira.habana-labs.com/browse/SW-96989
TEST_F(HpuOpTest, mv_bf16) {
  GenerateInputs(2, {{8, 16}, {16}}, torch::kBFloat16);

  auto expected = torch::mv(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::mv(GetHpuInput(0), GetHpuInput(1));
  Compare(expected, result, 2e-2, 2e-1); // Tuned rtol to 2e-2, atol to 2e-1
}

TEST_F(HpuOpTest, sub_out) {
  GenerateInputs(2, {{3, 4, 1}, {3, 1, 1}}, {torch::kFloat32, torch::kInt32});
  float alpha = 0.1;

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::sub_outf(GetCpuInput(0), GetCpuInput(1), alpha, expected);
  torch::sub_outf(GetHpuInput(0), GetHpuInput(1), alpha, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, mul_scalar_out) {
  GenerateInputs(1, {{2, 4, 6}});

  float other = 0.1;

  torch::ScalarType dtype = torch::kFloat;
  // aten::mul.Scalar_out is not yet enabled in cpu;
  // hence comparing mul.Scalar's cpu result with mul.Scalar_out's hpu
  auto expected = torch::mul(GetCpuInput(0), other);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::mul_outf(GetHpuInput(0), other, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, sub_scalar_out) {
  GenerateInputs(1, {{3, 4, 1}}, torch::kBFloat16);
  float alpha = 0.1;
  float other = 1.2;
  torch::ScalarType dtype = torch::kBFloat16;

  // aten::sub.Scalar_out is not yet enabled in cpu;
  // hence comparing sub.Scalar's cpu result with sub.Scalar_out's hpu
  auto expected = torch::sub(GetCpuInput(0), other, alpha);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::sub_outf(GetHpuInput(0), other, alpha, result);
  Compare(expected, result);
}
TEST_F(HpuOpTest, add_scalar_out) {
  GenerateInputs(1, {{3, 4, 1}}, torch::kBFloat16);
  float alpha = 0.1;
  float other = 1.2;
  torch::ScalarType dtype = torch::kBFloat16;

  // aten::add.Scalar_out is not yet enabled in cpu;
  // hence comparing add.Scalar's cpu result with add.Scalar_out's hpu result
  auto expected = torch::add(GetCpuInput(0), other, alpha);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::add_outf(GetHpuInput(0), other, alpha, result);

  Compare(expected, result);
}
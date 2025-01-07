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

TEST_F(HpuOpTest, signbit) {
  GenerateInputs(1);
  auto exp = torch::signbit(GetCpuInput(0));
  auto res = torch::signbit(GetHpuInput(0));
  Compare(exp, res, 0, 0);
}

TEST_F(HpuOpTest, signbit_bf16) {
  GenerateInputs(1, {{5, 6}}, {torch::kBFloat16});
  auto exp = torch::signbit(GetCpuInput(0));
  auto res = torch::signbit(GetHpuInput(0));
  Compare(exp, res, 0, 0);
}

TEST_F(HpuOpTest, signbit_int) {
  GenerateInputs(1, {{2, 3, 4, 5}}, {torch::kInt});
  auto exp = torch::signbit(GetCpuInput(0));
  auto res = torch::signbit(GetHpuInput(0));
  Compare(exp, res, 0, 0);
}

TEST_F(HpuOpTest, signbit_out) {
  GenerateInputs(1, torch::kFloat);
  torch::ScalarType dtype = torch::kBool;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::signbit_outf(GetCpuInput(0), expected);
  torch::signbit_outf(GetHpuInput(0), result);

  Compare(expected, result);
}

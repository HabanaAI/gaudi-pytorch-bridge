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

TEST_F(HpuOpTest, heaviside_f32) {
  GenerateInputs(2, {{2, 2}, {2, 2}}, {torch::kFloat, torch::kFloat});

  auto expected = torch::heaviside(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::heaviside(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}

TEST_F(HpuOpTest, heaviside_i32) {
  GenerateInputs(2, {{2, 2}, {2, 2}}, {torch::kInt, torch::kInt});

  auto expected = torch::heaviside(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::heaviside(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}

TEST_F(HpuOpTest, heaviside_out) {
  GenerateInputs(2, {{4, 3}, {1}}, {torch::kFloat, torch::kFloat});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::heaviside_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::heaviside_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, heaviside_) {
  GenerateInputs(2, {{2, 2}, {2, 2}}, {torch::kFloat, torch::kFloat});

  GetCpuInput(0).heaviside_(GetCpuInput(1));
  GetHpuInput(0).heaviside_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

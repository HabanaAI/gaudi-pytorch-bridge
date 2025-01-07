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

TEST_F(HpuOpTest, take) {
  GenerateInputs(1, {{4, 4}});
  auto float_cpu_input = GetCpuInput(0).clone();
  auto float_hpu_input = GetHpuInput(0).clone();

  GenerateIntInputs(1, {{4, 4}}, 0, 9);

  auto expected =
      torch::take(float_cpu_input, GetCpuInput(0).to(torch::kInt64));
  auto result = torch::take(float_hpu_input, GetHpuInput(0));
  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, take_out) {
  GenerateInputs(1, {{4, 4}}, {c10::ScalarType::BFloat16});
  auto float_cpu_input = GetCpuInput(0).clone();
  auto float_hpu_input = GetHpuInput(0).clone();

  GenerateIntInputs(1, {{4, 4}}, 0, 9);

  torch::ScalarType dtype = c10::ScalarType::BFloat16;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::take_outf(float_cpu_input, GetCpuInput(0).to(torch::kInt64), expected);
  torch::take_outf(float_hpu_input, GetHpuInput(0), result);
  Compare(expected, result, 0, 0);
}
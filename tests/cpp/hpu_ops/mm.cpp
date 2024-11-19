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

TEST_F(HpuOpTest, mm_float32) {
  GenerateInputs(2, {{2, 3}, {3, 3}});

  auto expected = torch::mm(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::mm(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}

/*
Note: Below testcase fails for BFloat16 with default tolerance
Issue raised : https://jira.habana-labs.com/browse/SW-71743
*/
TEST_F(HpuOpTest, mm_bfloat16) {
  torch::ScalarType dtype = torch::kBFloat16;

  GenerateInputs(2, {{30, 5}, {5, 25}}, dtype);

  auto expected = torch::mm(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::mm(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result, 1e-3, 1e-1);
}

TEST_F(HpuOpTest, mm_out_float32) {
  GenerateInputs(2, {{2, 3}, {3, 3}});

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::mm_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::mm_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}

/*
Note: Below testcase fails for BFloat16 with default tolerance
Issue raised : https://jira.habana-labs.com/browse/SW-71743
*/
TEST_F(HpuOpTest, mm_out_bfloat16) {
  torch::ScalarType dtype = torch::kBFloat16;

  GenerateInputs(2, {{30, 5}, {5, 25}}, dtype);

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::mm_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::mm_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result, 1e-03, 1e-01);
}
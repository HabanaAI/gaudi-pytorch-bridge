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

TEST_F(HpuOpTest, vdot) {
  GenerateInputs(2, {{12}, {12}});
  auto expected = torch::vdot(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::vdot(GetHpuInput(0), GetHpuInput(1));
  Compare(expected, result);
}

TEST_F(HpuOpTest, vdot_out) {
  GenerateInputs(2, {{10}, {10}});

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::vdot_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::vdot_outf(GetHpuInput(0), GetHpuInput(1), result);
  Compare(expected, result);
}
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

TEST_F(HpuOpTest, Relu) {
  GenerateInputs(1);

  auto expected = torch::relu(GetCpuInput(0));
  auto result = torch::relu(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, Relu_) {
  GenerateInputs(1);

  auto expected = GetCpuInput(0).relu();
  auto result = GetHpuInput(0).relu();

  Compare(expected, result);
}

TEST_F(HpuOpTest, Relu_out) {
  GenerateInputs(1, torch::kBFloat16);

  auto expected = torch::relu(GetCpuInput(0));
  auto result = torch::empty(0).to(torch::kBFloat16).to("hpu");
  torch::relu_outf(GetHpuInput(0), result);

  Compare(expected, result);
}

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

TEST_F(HpuOpTest, log10) {
  GenerateInputs(1, {{8, 24, 24, 3}});

  auto expected = torch::log10(GetCpuInput(0));
  auto result = torch::log10(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, log10_) {
  GenerateInputs(1, {{64, 64, 128}}, {torch::kBFloat16});

  GetCpuInput(0).log10_();
  GetHpuInput(0).log10_();

  Compare(GetCpuInput(0), GetHpuInput(0), 1e-02, 1e-02);
}

TEST_F(HpuOpTest, log10_out) {
  GenerateInputs(1, {{2, 64, 24, 12, 2}});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::log10_outf(GetCpuInput(0), expected);
  torch::log10_outf(GetHpuInput(0), result);

  Compare(expected, result);
}
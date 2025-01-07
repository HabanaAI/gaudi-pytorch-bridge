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

TEST_F(HpuOpTest, erfc) {
  GenerateInputs(1);

  auto expected = torch::erfc(GetCpuInput(0));
  auto result = torch::erfc(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, erfc_out) {
  GenerateInputs(1);
  torch::ScalarType dtype = torch::kFloat32;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::erfc_outf(GetCpuInput(0), expected);
  torch::erfc_outf(GetHpuInput(0), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, erfc_) {
  GenerateInputs(1);

  GetCpuInput(0).erfc_();
  GetHpuInput(0).erfc_();

  Compare(GetCpuInput(0), GetHpuInput(0));
}

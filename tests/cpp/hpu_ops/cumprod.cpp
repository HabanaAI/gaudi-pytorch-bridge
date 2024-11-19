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

TEST_F(HpuOpTest, cumprod) {
  GenerateInputs(1);
  int dim = -1;
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::cumprod(GetCpuInput(0), dim, dtype);
  auto result = torch::cumprod(GetHpuInput(0), dim, dtype);

  Compare(expected, result);
}

TEST_F(HpuOpTest, cumprod_) {
  GenerateInputs(1, torch::kInt);
  int dim = 1;

  GetCpuInput(0).cumprod_(dim);
  GetHpuInput(0).cumprod_(dim);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, cumprod_out) {
  GenerateInputs(1, torch::kInt);
  int dim = 2;
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::cumprod_outf(GetCpuInput(0), dim, dtype, expected);
  torch::cumprod_outf(GetHpuInput(0), dim, dtype, result);

  Compare(expected, result);
}

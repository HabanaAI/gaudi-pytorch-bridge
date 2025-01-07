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

TEST_F(HpuOpTest, glu) {
  GenerateInputs(1, {{4, 8, 16, 32}});

  auto expected = torch::glu(GetCpuInput(0), /*dim*/ -3);
  auto result = torch::glu(GetHpuInput(0), /*dim*/ -3);

  Compare(expected, result);
}

TEST_F(HpuOpTest, glu_out) {
  GenerateInputs(1, {{4, 8, 16, 32, 32}});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::glu_out(expected, GetCpuInput(0), /*dim*/ 2);
  torch::glu_out(result, GetHpuInput(0), /*dim*/ 2);

  Compare(expected, result);
}

TEST_F(HpuOpTest, glu_bwd) {
  GenerateInputs(2, {{4, 8, 16, 32}, {2, 8, 16, 32}});

  torch::ScalarType dtype = torch::kFloat;

  auto expected =
      torch::glu_backward(GetCpuInput(1), GetCpuInput(0), /*dim*/ 0);
  auto result = torch::glu_backward(GetHpuInput(1), GetHpuInput(0), /*dim*/ 0);

  Compare(expected, result);
}

TEST_F(HpuOpTest, glu_bwd_out) {
  GenerateInputs(2, {{4, 8, 16, 32, 32}, {4, 8, 16, 16, 32}});

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::glu_backward_out(expected, GetCpuInput(1), GetCpuInput(0), /*dim*/ 3);
  torch::glu_backward_out(result, GetHpuInput(1), GetHpuInput(0), /*dim*/ 3);

  Compare(expected, result);
}
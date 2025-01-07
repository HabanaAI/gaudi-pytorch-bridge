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

TEST_F(HpuOpTest, flatten_dense_tensors_2_inputs) {
  GenerateInputs(2, {{64}, {64}}, torch::kBFloat16);

  torch::ScalarType dtype = torch::kBFloat16;

  auto expected =
      torch::flatten_dense_tensors({GetCpuInput(0), GetCpuInput(1)});
  auto result = torch::flatten_dense_tensors({GetHpuInput(0), GetHpuInput(1)});

  // As no computation involved, tolerance is set to 0
  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, flatten_dense_tensors_3_inputs) {
  GenerateInputs(3, {{8, 16}, {4, 8, 2}, {2}});

  auto expected = torch::flatten_dense_tensors(
      {GetCpuInput(0), GetCpuInput(1), GetCpuInput(2)});
  auto result = torch::flatten_dense_tensors(
      {GetHpuInput(0), GetHpuInput(1), GetHpuInput(2)});

  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, flatten_dense_tensors_4_inputs) {
  GenerateInputs(4, {{8, 16, 2}, {2, 24}, {16}, {2, 5}});

  auto expected = torch::flatten_dense_tensors(
      {GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), GetCpuInput(3)});
  auto result = torch::flatten_dense_tensors(
      {GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), GetHpuInput(3)});

  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, flatten_dense_tensors_5_inputs) {
  GenerateInputs(5, {{8, 16, 2}, {2, 24}, {16}, {2, 5}, {2}}, torch::kBFloat16);

  auto expected = torch::flatten_dense_tensors(
      {GetCpuInput(0),
       GetCpuInput(1),
       GetCpuInput(2),
       GetCpuInput(3),
       GetCpuInput(4)});
  auto result = torch::flatten_dense_tensors(
      {GetHpuInput(0),
       GetHpuInput(1),
       GetHpuInput(2),
       GetHpuInput(3),
       GetHpuInput(4)});

  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, flatten_dense_tensors_1_input) {
  GenerateInputs(1, {{2, 16, 2, 2, 4}}, torch::kBFloat16);

  auto expected = torch::flatten_dense_tensors({GetCpuInput(0)});
  auto result = torch::flatten_dense_tensors({GetHpuInput(0)});

  Compare(expected, result, 0, 0);
}

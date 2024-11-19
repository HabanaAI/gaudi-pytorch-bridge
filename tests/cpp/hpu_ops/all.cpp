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

TEST_F(HpuOpTest, all_dim_out) {
  GenerateInputs(1, {{4, 8, 16, 32}}, {torch::kBool});
  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::all_out(expected, GetCpuInput(0), /*dim*/ 1, /*keepdim*/ true);
  torch::all_out(result, GetHpuInput(0), /*dim*/ 1, /*keepdim*/ true);

  Compare(expected, result);
}

TEST_F(HpuOpTest, all_keepdim_false_out) {
  GenerateInputs(1, {{12, 5, 6, 3}}, {torch::kBool});
  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::all_out(expected, GetCpuInput(0), /*dim*/ 0, /*keepdim*/ false);
  torch::all_out(result, GetHpuInput(0), /*dim*/ 0, /*keepdim*/ false);

  Compare(expected, result);
}

TEST_F(HpuOpTest, all_neg_dim_out) {
  GenerateInputs(1, {{3, 11, 2, 4}}, {torch::kBool});
  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::all_out(expected, GetCpuInput(0), /*dim*/ -1, /*keepdim*/ true);
  torch::all_out(result, GetHpuInput(0), /*dim*/ -1, /*keepdim*/ true);

  Compare(expected, result);
}

TEST_F(HpuOpTest, all_out) {
  GenerateInputs(1, {{4, 8, 16}}, {torch::kBool});
  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::all_out(expected, GetCpuInput(0));
  torch::all_out(result, GetHpuInput(0));

  Compare(expected, result);
}

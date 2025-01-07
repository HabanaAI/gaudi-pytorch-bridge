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

TEST_F(HpuOpTest, any_out) {
  GenerateInputs(1, {{4, 8, 16, 32}}, {torch::kBool});
  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::any_out(expected, GetCpuInput(0));
  torch::any_out(result, GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, any_keepdim) {
  GenerateInputs(1, {{4, 8, 16, 32}});

  auto expected = torch::any(GetCpuInput(0), 0, true);
  auto result = torch::any(GetHpuInput(0), 0, true);

  Compare(expected, result);
}

TEST_F(HpuOpTest, any_bf16) {
  GenerateInputs(1, {{4, 8, 16}}, {torch::kBFloat16});

  auto expected = torch::any(GetCpuInput(0), 1, false);
  auto result = torch::any(GetHpuInput(0), 1, false);

  Compare(expected, result);
}

// input inf
TEST_F(HpuOpTest, any_inf) {
  float pos_inf = std::numeric_limits<int>::infinity();
  float neg_inf = -std::numeric_limits<int>::infinity();
  auto tensor1 = torch::tensor({pos_inf, neg_inf});
  auto tensor2 = torch::tensor({pos_inf, neg_inf}, "hpu");

  auto expected = torch::any(tensor1, 0, true);
  auto result = torch::any(tensor2, 0, true);

  Compare(expected, result);
}
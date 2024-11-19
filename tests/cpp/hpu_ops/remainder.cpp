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

TEST_F(HpuOpTest, remainder) {
  GenerateInputs(2, {{2, 1}, {2, 3}}, {torch::kLong, torch::kFloat});
  auto exp = torch::remainder(GetCpuInput(0), GetCpuInput(1));
  auto res = torch::remainder(GetHpuInput(0), GetHpuInput(1));

  Compare(exp, res);
}

TEST_F(HpuOpTest, remainderFloat) {
  GenerateInputs(2, {{2, 1}, {2, 3}}, {torch::kFloat, torch::kFloat});
  auto exp = torch::remainder(GetCpuInput(0), GetCpuInput(1));
  auto res = torch::remainder(GetHpuInput(0), GetHpuInput(1));

  Compare(exp, res);
}

TEST_F(HpuOpTest, remainder_scalar) {
  GenerateIntInputs(1, {{2, 3, 3}}, -10000, 10000);
  auto exp = torch::remainder(GetCpuInput(0), 25);
  auto res = torch::remainder(GetHpuInput(0), 25);

  Compare(exp, res);
}

TEST_F(HpuOpTest, remainder_scalar_tensor) {
  GenerateIntInputs(1, {{2, 3, 3}}, -10000, 10000);
  auto exp = torch::remainder(25, GetCpuInput(0));
  auto res = torch::remainder(25, GetHpuInput(0));

  Compare(exp, res);
}

TEST_F(HpuOpTest, remainder_scalar_out) {
  GenerateIntInputs(1, {{2, 3, 3}}, -10000, 10000);
  c10::ScalarType dtype = torch::kFloat;

  auto exp = torch::empty(0, dtype);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::remainder_outf(GetCpuInput(0), 4.2, exp);
  torch::remainder_outf(GetHpuInput(0), 4.2, res);

  Compare(exp, res);
}

TEST_F(HpuOpTest, remainder_tensor_out) {
  GenerateInputs(2, {{2, 3}, {3}}, {torch::kLong, torch::kDouble});
  c10::ScalarType dtype = torch::kDouble;

  auto exp = torch::empty(0, dtype);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::remainder_outf(GetCpuInput(0), GetCpuInput(1), exp);
  torch::remainder_outf(GetHpuInput(0), GetHpuInput(1), res);

  Compare(exp, res);
}

TEST_F(HpuOpTest, remainder_) {
  GenerateIntInputs(2, {{3, 3}, {3, 1}}, -30, 10);
  auto exp = GetCpuInput(0).remainder_(GetCpuInput(1));
  auto res = GetHpuInput(0).remainder_(GetHpuInput(1));

  Compare(exp, res);
}

TEST_F(HpuOpTest, remainder__scalar) {
  GenerateInputs(1);
  auto exp = GetCpuInput(0).remainder_(-2.3);
  auto res = GetHpuInput(0).remainder_(-2.3);

  Compare(exp, res);
}

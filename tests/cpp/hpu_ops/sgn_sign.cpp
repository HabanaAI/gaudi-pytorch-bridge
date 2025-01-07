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

TEST_F(HpuOpTest, sgn) {
  GenerateInputs(1);
  auto exp = torch::sgn(GetCpuInput(0));
  auto res = torch::sgn(GetHpuInput(0));
  Compare(exp, res, 0, 0);
}

TEST_F(HpuOpTest, sgn_bf16) {
  GenerateInputs(1, {{5, 6}}, {torch::kBFloat16});
  auto exp = torch::sgn(GetCpuInput(0));
  auto res = torch::sgn(GetHpuInput(0));
  Compare(exp, res, 0, 0);
}

TEST_F(HpuOpTest, sgn_inplace) {
  GenerateInputs(1, torch::kFloat);
  auto expected = GetCpuInput(0);
  auto result = GetHpuInput(0);

  expected = torch::sgn(expected);
  result = torch::sgn(result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, sgn_out) {
  GenerateInputs(1, torch::kFloat);
  auto expected = torch::empty(0);
  auto result = torch::empty(0, torch::TensorOptions().device("hpu"));

  torch::sgn_outf(GetCpuInput(0), expected);
  torch::sgn_outf(GetHpuInput(0), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, sign) {
  GenerateInputs(1);
  auto exp = torch::sign(GetCpuInput(0));
  auto res = torch::sign(GetHpuInput(0));
  Compare(exp, res, 0, 0);
}

TEST_F(HpuOpTest, sign_bf16) {
  GenerateInputs(1, {{5, 6}}, {torch::kBFloat16});
  auto exp = torch::sign(GetCpuInput(0));
  auto res = torch::sign(GetHpuInput(0));
  Compare(exp, res, 0, 0);
}

TEST_F(HpuOpTest, sign_inplace) {
  GenerateInputs(1, torch::kFloat);
  auto expected = GetCpuInput(0);
  auto result = GetHpuInput(0);

  expected = torch::sign(expected);
  result = torch::sign(result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, sign_out) {
  GenerateInputs(1, torch::kFloat);
  auto expected = torch::empty(0);
  auto result = torch::empty(0, torch::TensorOptions().device("hpu"));

  torch::sign_outf(GetCpuInput(0), expected);
  torch::sign_outf(GetHpuInput(0), result);

  Compare(expected, result);
}

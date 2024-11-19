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

TEST_F(HpuOpTest, fmod) {
  // TODO: avoid 0 in 2nd input to avoid divide by zero exception in cpu run
  // GenerateInputs(2, {{2, 3}, {2, 3}}, {torch::kInt, torch::kFloat});
  GenerateInputs(2, {{2, 1}, {2, 3}}, {torch::kLong, torch::kFloat});
  // GenerateInputs(2, {{2, 1}, {2, 100}}, {torch::kDouble, torch::kLong});
  auto exp = torch::fmod(GetCpuInput(0), GetCpuInput(1));
  auto res = torch::fmod(GetHpuInput(0), GetHpuInput(1));

  Compare(exp, res);
}

TEST_F(HpuOpTest, fmod_scalar) {
  GenerateIntInputs(1, {{2, 3, 3}}, -10000, 10000);
  auto exp = torch::fmod(GetCpuInput(0), 25);
  auto res = torch::fmod(GetHpuInput(0), 25);

  Compare(exp, res);
}

TEST_F(HpuOpTest, fmod_scalar_out) {
  GenerateIntInputs(1, {{2, 3, 3}}, -10000, 10000);
  c10::ScalarType dtype = torch::kFloat;

  auto exp = torch::empty(0, dtype);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::fmod_outf(GetCpuInput(0), 4.2, exp);
  torch::fmod_outf(GetHpuInput(0), 4.2, res);

  Compare(exp, res);
}

TEST_F(HpuOpTest, fmod_tensor_out) {
  GenerateInputs(2, {{2, 3}, {3}}, {torch::kLong, torch::kDouble});
  c10::ScalarType dtype = torch::kDouble;

  auto exp = torch::empty(0, dtype);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::fmod_outf(GetCpuInput(0), GetCpuInput(1), exp);
  torch::fmod_outf(GetHpuInput(0), GetHpuInput(1), res);

  Compare(exp, res);
}

TEST_F(HpuOpTest, fmod_) {
  GenerateIntInputs(2, {{2, 3, 3}, {2, 3, 1}}, -30, 10);
  auto exp = GetCpuInput(0).fmod_(GetCpuInput(1));
  auto res = GetHpuInput(0).fmod_(GetHpuInput(1));

  Compare(exp, res);
}

TEST_F(HpuOpTest, fmod__scalar) {
  GenerateInputs(1);
  auto exp = GetCpuInput(0).fmod_(-2.3);
  auto res = GetHpuInput(0).fmod_(-2.3);

  Compare(exp, res);
}

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

class DivScalarHpuOpTest : public HpuOpTestUtil {};

TEST_F(DivScalarHpuOpTest, div_scalar_int_int) {
  GenerateInputs(1, {{4, 5}}, torch::kInt);
  auto expected = torch::div(GetCpuInput(0), 11);
  auto result = torch::div(GetHpuInput(0), 11);
  Compare(expected, result);
}

TEST_F(DivScalarHpuOpTest, div_scalar_float_int) {
  GenerateInputs(1, {{4, 5}}, torch::kFloat);
  auto expected = torch::div(GetCpuInput(0), 11);
  auto result = torch::div(GetHpuInput(0), 11);
  Compare(expected, result);
}

TEST_F(DivScalarHpuOpTest, div_scalar_bfloat_int) {
  GenerateInputs(1, {{4, 5}}, torch::kBFloat16);
  auto expected = torch::div(GetCpuInput(0), 11);
  auto result = torch::div(GetHpuInput(0), 11);
  Compare(expected, result);
}

TEST_F(DivScalarHpuOpTest, div_scalar_int_float) {
  GenerateInputs(1, {{4, 5}}, torch::kInt);
  auto expected = torch::div(GetCpuInput(0), 1.1);
  auto result = torch::div(GetHpuInput(0), 1.1);
  Compare(expected, result);
}

TEST_F(DivScalarHpuOpTest, div_scalar_bfloat_float) {
  GenerateInputs(1, {{4, 5}}, torch::kBFloat16);
  auto expected = torch::div(GetCpuInput(0), 1.1);
  auto result = torch::div(GetHpuInput(0), 1.1);
  Compare(expected, result, 2e-2, 2e-2);
}

TEST_F(DivScalarHpuOpTest, div_scalar_float_float) {
  GenerateInputs(1, {{4, 5}}, torch::kFloat);
  auto expected = torch::div(GetCpuInput(0), 1.1);
  auto result = torch::div(GetHpuInput(0), 1.1);
  Compare(expected, result);
}
// inplace scalar ops
TEST_F(DivScalarHpuOpTest, div_scalar_float_int_) {
  GenerateInputs(1, {{4, 5}}, torch::kFloat);
  GetCpuInput(0).div_(11);
  GetHpuInput(0).div_(11);
  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(DivScalarHpuOpTest, div_scalar_bfloat_int_) {
  GenerateInputs(1, {{4, 5}}, torch::kBFloat16);
  GetCpuInput(0).div_(11);
  GetHpuInput(0).div_(11);
  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(DivScalarHpuOpTest, div_scalar_bfloat_float_) {
  GenerateInputs(1, {{4, 5}}, torch::kBFloat16);
  GetCpuInput(0).div_(1.1);
  GetHpuInput(0).div_(1.1);
  Compare(GetCpuInput(0), GetHpuInput(0), 2e-2, 2e-2);
}

TEST_F(DivScalarHpuOpTest, div_scalar_float_float_) {
  GenerateInputs(1, {{4, 5}}, torch::kFloat);
  GetCpuInput(0).div_(1.1);
  GetHpuInput(0).div_(1.1);
  Compare(GetCpuInput(0), GetHpuInput(0));
}
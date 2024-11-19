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

TEST_F(HpuOpTest, fill_tensor_inplace_float) {
  GenerateInputs(2, {{3, 2, 3}, {}});

  GetCpuInput(0).fill_(GetCpuInput(1));
  GetHpuInput(0).fill_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, fill_tensor_inplace_bfloat) {
  GenerateInputs(2, {{4, 5, 1, 2}, {}}, torch::kBFloat16);

  GetCpuInput(0).fill_(GetCpuInput(1));
  GetHpuInput(0).fill_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, fill_tensor_inplace_int) {
  GenerateInputs(2, {{6, 8}, {}}, torch::kInt32);

  GetCpuInput(0).fill_(GetCpuInput(1));
  GetHpuInput(0).fill_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

// Fill Out variants are not supported in CPU
// so used usual variants for reference

TEST_F(HpuOpTest, fillt_int) {
  auto dtype = torch::kInt32;
  GenerateInputs(2, {{6, 8}, {}}, dtype);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  auto exp = torch::fill(GetCpuInput(0), GetCpuInput(1));
  torch::fill_outf(GetHpuInput(0), GetHpuInput(1), res);

  Compare(exp, res);
}

TEST_F(HpuOpTest, fills_f32) {
  auto dtype = torch::kFloat;
  GenerateInputs(1, {{2, 6, 8}}, dtype);
  at::Scalar other = 2;
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  auto exp = torch::fill(GetCpuInput(0), other);
  torch::fill_outf(GetHpuInput(0), other, res);

  Compare(exp, res);
}

TEST_F(HpuOpTest, fillt_i16) {
  auto dtype = torch::kInt16;
  GenerateInputs(2, {{2, 4, 6, 8}, {}}, dtype);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  auto exp = torch::fill(GetCpuInput(0), GetCpuInput(1));
  torch::fill_outf(GetHpuInput(0), GetHpuInput(1), res);

  Compare(exp, res);
}

TEST_F(HpuOpTest, fills_i8) {
  auto dtype = torch::kInt8;
  GenerateInputs(1, {{5, 2, 4, 6, 8}}, dtype);
  at::Scalar other = 3.0;
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  auto exp = torch::fill(GetCpuInput(0), other);
  torch::fill_outf(GetHpuInput(0), other, res);
  Compare(exp, res);
}

TEST_F(HpuOpTest, fill_bool) {
  auto dtype = torch::kBFloat16;
  GenerateInputs(1, {{5, 2, 4, 6, 8}}, dtype);
  bool other = 1;
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  GetCpuInput(0).fill_(other);
  torch::fill_outf(GetHpuInput(0), other, res);
  Compare(GetCpuInput(0), res);
}
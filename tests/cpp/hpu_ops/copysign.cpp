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

#include "../utils/device_type_util.h"
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, copysign_out_tensor) {
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(at::kFloat));
  GenerateInputs(2);

  auto expected = torch::empty(0);
  auto result =
      torch::empty(0, torch::TensorOptions(torch::kFloat).device("hpu"));

  torch::copysign_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::copysign_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, copysign_out_bf16) {
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(at::kFloat));
  GenerateInputs(2, {torch::kBFloat16, torch::kFloat});

  auto expected = torch::empty(0, torch::kFloat);
  auto result =
      torch::empty(0, torch::TensorOptions(torch::kFloat).device("hpu"));

  torch::copysign_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::copysign_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, copysign_out_int) {
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(at::kBFloat16));
  GenerateInputs(2, torch::kInt);

  auto expected = torch::empty(0, torch::kBFloat16);
  auto result =
      torch::empty(0, torch::TensorOptions(torch::kBFloat16).device("hpu"));

  torch::copysign_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::copysign_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, copysign_out_z) {
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(at::kFloat));
  GenerateInputs(1, {{2, 3, 5}});

  auto zeros = torch::zeros({2, 3, 5}, "hpu");
  auto z_cpu = torch::zeros({2, 3, 5});

  auto expected = torch::empty({2, 3, 5});
  auto result = torch::empty(
      {2, 3, 5}, torch::TensorOptions(torch::kFloat).device("hpu"));
  torch::copysign_outf(GetCpuInput(0), z_cpu, expected);
  torch::copysign_outf(GetHpuInput(0), zeros, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, copysign_out_bc) {
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(at::kFloat));
  GenerateInputs(2, {{1, 3, 4}, {2, 3, 4}});

  auto expected = torch::empty(0);
  auto result =
      torch::empty(0, torch::TensorOptions(torch::kFloat).device("hpu"));

  torch::copysign_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::copysign_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, copysign_scalar) {
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(at::kFloat));
  GenerateInputs(1);

  auto other = GenerateScalar<int>(0, 0);
  auto expected = torch::copysign(GetCpuInput(0), other);
  auto result = torch::copysign(GetHpuInput(0), other);

  Compare(expected, result);
}

TEST_F(HpuOpTest, copysign_scalar_) {
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(at::kFloat));
  GenerateInputs(1);

  auto other = GenerateScalar<float>(0.3, 0.5);
  GetCpuInput(0).copysign_(other);
  GetHpuInput(0).copysign_(other);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, copysign_scalar_bf16) {
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(at::kFloat));
  GenerateInputs(1, torch::kBFloat16);
  auto other = GenerateScalar<int>(-3, 3);

  auto expected = torch::copysign(GetCpuInput(0), other);
  auto result = torch::copysign(GetHpuInput(0), other);

  Compare(expected, result);
}

TEST_F(HpuOpTest, copysign_scalar_int) {
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(at::kBFloat16));
  GenerateInputs(1, torch::kInt);
  auto other = GenerateScalar<int>();

  auto expected = torch::copysign(GetCpuInput(0), other);
  auto result = torch::copysign(GetHpuInput(0), other);

  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, copysign_bc) {
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(at::kFloat));
  GenerateInputs(2, {torch::kBFloat16, torch::kBFloat16});

  auto expected = torch::copysign(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::copysign(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}

TEST_F(HpuOpTest, copysign_) {
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(at::kFloat));
  GenerateInputs(2);

  GetCpuInput(0).copysign_(GetCpuInput(1));
  GetHpuInput(0).copysign_(GetCpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, copysign_scalar_out) {
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(at::kFloat));
  GenerateInputs(1);

  auto other = GenerateScalar<float>(-1.1, -0.5);
  auto expected = torch::empty(0);
  auto result =
      torch::empty(0, torch::TensorOptions(torch::kFloat).device("hpu"));

  torch::copysign_outf(GetCpuInput(0), other, expected);
  torch::copysign_outf(GetHpuInput(0), other, result);

  Compare(expected, result);
}

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

#include <gtest/gtest.h>
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, pow_tensor_tensor_out) {
  GenerateInputs(2, {torch::kBFloat16, torch::kF32});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::pow_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::pow_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, pow_scalar_) {
  GenerateInputs(1);

  GetCpuInput(0).pow_(3);
  GetHpuInput(0).pow_(3);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, pow_scalar) {
  GenerateInputs(1);

  auto expected = torch::pow(GetCpuInput(0), 2);
  auto result = torch::pow(GetHpuInput(0), 2);
  Compare(expected, result);
}
TEST_F(HpuOpTest, pow_out_non_hpu_out_tensor) {
  GenerateInputs(2);
  auto out = torch::empty(0);
  EXPECT_THROW(
      torch::pow_outf(GetHpuInput(0), GetHpuInput(1), out), c10::Error);
}

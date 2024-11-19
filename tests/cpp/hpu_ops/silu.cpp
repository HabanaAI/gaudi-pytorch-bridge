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

TEST_F(HpuOpTest, silu_) {
  GenerateInputs(1, torch::kBFloat16);

  auto expected = torch::silu_(GetCpuInput(0));
  auto result = torch::silu_(GetHpuInput(0));

  /*Bug ticket for high tolerance used:
  https://jira.habana-labs.com/browse/PO-510 */

  Compare(expected, result, 0.32, 1e-3);
}

TEST_F(HpuOpTest, silu) {
  GenerateInputs(1);

  auto expected = torch::silu(GetCpuInput(0));
  auto result = torch::silu(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, silu_out) {
  GenerateInputs(1);

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::silu_outf(GetCpuInput(0), expected);
  torch::silu_outf(GetHpuInput(0), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, silu_bwd) {
  GenerateInputs(2);

  auto expected = torch::silu_backward(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::silu_backward(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}

TEST_F(HpuOpTest, silu_bwd_bfloat16) {
  GenerateInputs(2, torch::kBFloat16);

  auto expected = torch::silu_backward(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::silu_backward(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result, 0.1, 0.1);
}

TEST_F(HpuOpTest, silu_bwd_out) {
  GenerateInputs(2);

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  expected =
      torch::silu_backward_outf(GetCpuInput(0), GetCpuInput(1), expected);
  result = torch::silu_backward_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}
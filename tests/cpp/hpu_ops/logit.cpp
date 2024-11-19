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

TEST_F(HpuOpTest, logit) {
  GenerateInputs(1);

  auto expected = torch::logit(GetCpuInput(0), /*eps*/ 1e-06);
  auto result = torch::logit(GetHpuInput(0), /*eps*/ 1e-06);

  Compare(expected, result);
}

/**
Below Testcase fails with default tolerance for BF16
Issue raised:https://jira.habana-labs.com/browse/SW-68424
**/
TEST_F(HpuOpTest, logit_) {
  GenerateInputs(1, torch::kBFloat16);

  GetCpuInput(0).logit_(/*eps*/ 5e-04);
  GetHpuInput(0).logit_(/*eps*/ 5e-04);

  Compare(GetCpuInput(0), GetHpuInput(0), 1e-03, 4e-02);
}

TEST_F(HpuOpTest, logit_out) {
  GenerateInputs(1);

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::logit_outf(GetCpuInput(0), /*eps*/ c10::nullopt, expected);
  torch::logit_outf(GetHpuInput(0), /*eps*/ c10::nullopt, result);

  Compare(expected, result);
}

/**
Below Testcase fails with default tolerance for BF16
Issue raised:https://jira.habana-labs.com/browse/SW-68424
**/
TEST_F(HpuOpTest, logit_backward) {
  const std::vector<int64_t> size = {7, 6};
  GenerateInputs(2, {size, size}, torch::kBFloat16);

  torch::ScalarType dtype = torch::kBFloat16;

  auto expected =
      torch::logit_backward(GetCpuInput(1), GetCpuInput(0), /*eps*/ 4e-10);
  auto result =
      torch::logit_backward(GetHpuInput(1), GetHpuInput(0), /*eps*/ 4e-10);

  Compare(expected, result, 1e-03, 4e-02);
}

TEST_F(HpuOpTest, logit_backward_out) {
  const std::vector<int64_t> size = {5, 3, 8};
  GenerateInputs(2, {size, size});

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::logit_backward_outf(
      GetCpuInput(1), GetCpuInput(0), /*eps*/ 7e-05, expected);
  torch::logit_backward_outf(
      GetHpuInput(1), GetHpuInput(0), /*eps*/ 7e-05, result);

  Compare(expected, result);
}

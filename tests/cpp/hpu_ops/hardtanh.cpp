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

TEST_F(HpuOpTest, hardtanh_out) {
  GenerateInputs(1);
  int val = GenerateScalar();
  int max_val = (val > 0) ? val : -1 * val;
  int min_val = -1 * max_val;

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::hardtanh_outf(GetCpuInput(0), min_val, max_val, expected);
  torch::hardtanh_outf(GetHpuInput(0), min_val, max_val, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, hardtanh_bwd_out) {
  GenerateInputs(2);
  int val = GenerateScalar();
  int max_val = (val > 0) ? val : -1 * val;
  int min_val = -1 * max_val;

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::hardtanh_backward_outf(
      GetCpuInput(0), GetCpuInput(1), min_val, max_val, expected);
  torch::hardtanh_backward_outf(
      GetHpuInput(0), GetHpuInput(1), min_val, max_val, result);

  Compare(expected, result);
}